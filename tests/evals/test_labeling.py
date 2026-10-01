import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from grasp_agents.evals import (
    Dataset,
    DatasetError,
    EvalContext,
    EvaluationRun,
    Example,
    LocalRunStore,
    evaluate,
    evaluator,
    import_labels,
    judge_validation,
    judged_outputs,
    sample_for_labeling,
)
from grasp_agents.evals.cli import main
from grasp_agents.evals.labeling import (
    annotation_value,
    known_splits,
    label_split,
    merge_labels,
    write_records,
)
from grasp_agents.evals.phoenix import PhoenixClient
from grasp_agents.evals.phoenix.annotations import human_annotations

type Ctx = EvalContext[int, str, Any]


@evaluator(name="judge", annotator="LLM")
def judge(ctx: Ctx) -> bool:
    return ctx.input % 2 == 0


@evaluator(name="check")
def check(ctx: Ctx) -> bool:
    return ctx.input % 3 == 0


async def _describe(x: int) -> str:
    if x == 5:
        raise RuntimeError("failed on 5")
    return f"answer {x}"


def _numbers() -> Dataset[int, Any]:
    return Dataset(
        [
            Example(
                id=f"n{i}",
                input=i,
                metadata={"topic": "a" if i < 6 else "b"},
                splits=["test"] if i >= 10 else ["dev"],
            )
            for i in range(12)
        ],
        name="numbers",
    )


async def _run(store: LocalRunStore, repetitions: int = 1) -> EvaluationRun:
    return await evaluate(
        _describe,
        _numbers(),
        [judge, check],
        repetitions=repetitions,
        sealed_splits=["test"],
        store=store,
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalRunStore:
    return LocalRunStore(tmp_path / "evals")


def _examples(records: list[dict[str, Any]]) -> list[str]:
    return [r["metadata"]["example_id"] for r in records]


class TestSampling:
    @pytest.mark.asyncio
    async def test_never_exports_sealed_or_failed_trials_and_dedupes_outputs(
        self, store: LocalRunStore
    ) -> None:
        run = await _run(store, repetitions=3)
        records = sample_for_labeling(run, 100, score="judge")
        chosen = _examples(records)
        # n10/n11 are sealed, n5 failed; identical repetitions appear once.
        assert sorted(chosen) == sorted(f"n{i}" for i in range(10) if i != 5)
        for record in records:
            assert record["reference"] is None
            assert record["metadata"]["score"] == "judge"
            assert record["metadata"]["run_id"] == run.id
            assert set(record["input"]) == {"input", "output", "metadata"}
            # Blind: no verdicts in the request.
            assert "judge" not in json.dumps(record["input"])

    @pytest.mark.asyncio
    async def test_disagreements_first_then_spread_over_verdicts(
        self, store: LocalRunStore
    ) -> None:
        run = await _run(store)
        records = sample_for_labeling(run, 8, score="judge", against="check")
        chosen = [int(e[1:]) for e in _examples(records)]
        disagree = [x for x in range(10) if x != 5 and (x % 2 == 0) != (x % 3 == 0)]
        assert sorted(chosen[: len(disagree)]) == disagree
        rest = chosen[len(disagree) :]
        # The other three alternate between the judge's verdicts.
        assert len(rest) == 3
        assert {x % 2 == 0 for x in rest} == {False, True}

        balanced = sample_for_labeling(run, 4, score="judge", strata="topic")
        picks = [int(e[1:]) for e in _examples(balanced)]
        groups = {(x % 2 == 0, x < 6) for x in picks}
        assert len(groups) == 4

    @pytest.mark.asyncio
    async def test_exclude_seed_and_stable_splits(self, store: LocalRunStore) -> None:
        run = await _run(store)
        first = sample_for_labeling(run, 3, score="judge", seed=1)
        again = sample_for_labeling(run, 3, score="judge", seed=1)
        assert first == again
        rest = sample_for_labeling(
            run, 100, score="judge", exclude=[r["id"] for r in first]
        )
        assert not {r["id"] for r in first} & {r["id"] for r in rest}
        assert len(first) + len(rest) == 9
        for record in [*first, *rest]:
            example_id = record["metadata"]["example_id"]
            assert record["splits"] == [label_split(example_id, 0.4)]
        assert {
            s for r in sample_for_labeling(run, 9, test_share=0.0) for s in r["splits"]
        } == {"dev"}
        assert {
            s for r in sample_for_labeling(run, 9, test_share=1.0) for s in r["splits"]
        } == {"test"}

    def test_rejects_bad_arguments(self) -> None:
        run = EvaluationRun.model_construct(trials=[], examples=[])
        with pytest.raises(ValueError, match="n must be"):
            sample_for_labeling(run, 0)
        with pytest.raises(ValueError, match="test_share"):
            sample_for_labeling(run, 1, test_share=1.5)

    @pytest.mark.asyncio
    async def test_judged_outputs_are_typed_items(self, store: LocalRunStore) -> None:
        run = await _run(store)
        items = judged_outputs(run, input_type=int, output_type=str)
        assert len(items) == 9
        first = items[0]
        assert first.input.input == 0
        assert first.input.output == "answer 0"
        assert first.metadata["example_id"] == "n0"


def _filled(records: list[dict[str, Any]], labels: list[Any]) -> list[dict[str, Any]]:
    return [{**r, "reference": label} for r, label in zip(records, labels, strict=True)]


class TestImport:
    @pytest.mark.asyncio
    async def test_merges_labels_and_reports_conflicts(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await _run(store)
        requests = sample_for_labeling(run, 4, score="judge")
        into = tmp_path / "labels.jsonl"
        filled = write_records(
            _filled(requests, [True, False, None, True]), tmp_path / "round1.jsonl"
        )
        first = import_labels([filled], into, labeler="ana")
        assert (first.added, first.unlabeled, first.total) == (3, 1, 3)
        stored = Dataset.load(into)
        # Labels and provenance are kept per score.
        assert {json.dumps(e.reference) for e in stored} == {
            '{"judge": true}',
            '{"judge": false}',
        }
        assert all(e.metadata["labeler"] == {"judge": "ana"} for e in stored)
        assert all(set(e.metadata["labeled_at"]) == {"judge"} for e in stored)

        again = import_labels([filled], into, labeler="ben")
        assert (again.added, again.unchanged, again.conflicts) == (0, 3, [])

        changed = write_records(
            _filled(requests, [False, False, True, True]), tmp_path / "round2.jsonl"
        )
        clash = import_labels([changed], into, labeler="ben")
        assert (clash.added, clash.unchanged) == (1, 3)
        assert [c["id"] for c in clash.conflicts] == [requests[0]["id"]]
        assert clash.conflicts[0]["scores"] == ["judge"]
        assert Dataset.load(into)[requests[0]["id"]].reference == {"judge": True}

        replaced = import_labels([changed], into, labeler="cy", replace=True)
        assert replaced.updated == 1
        relabeled = Dataset.load(into)[requests[0]["id"]]
        assert relabeled.reference == {"judge": False}
        assert relabeled.metadata["labeler"] == {"judge": "cy"}

    def test_several_scores_merge_per_score(self, tmp_path: Path) -> None:
        record = {"id": "x", "input": {"input": 1, "output": "o"}}
        into = tmp_path / "labels.jsonl"
        merge_labels([{**record, "reference": {"clarity": True}}], into)
        result = merge_labels(
            [{**record, "reference": {"clarity": False, "tone": "kind"}}], into
        )
        assert result.updated == 1
        assert result.conflicts[0]["scores"] == ["clarity"]
        assert Dataset.load(into)["x"].reference == {"clarity": True, "tone": "kind"}

    def test_rejects_bad_labels_and_changed_outputs(self, tmp_path: Path) -> None:
        into = tmp_path / "labels.jsonl"
        with pytest.raises(DatasetError, match="a label is"):
            merge_labels([{"id": "x", "input": {}, "reference": [1, 2]}], into)
        merge_labels([{"id": "x", "input": {"output": "a"}, "reference": True}], into)
        with pytest.raises(DatasetError, match="different output"):
            merge_labels(
                [{"id": "x", "input": {"output": "b"}, "reference": True}], into
            )

    @pytest.mark.asyncio
    async def test_imported_labels_validate_the_judge(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await _run(store)
        requests = sample_for_labeling(run, 9, score="judge", test_share=0.0)
        # People agree with the judge except on multiples of three.
        labels = [
            (int(r["metadata"]["example_id"][1:]) % 2 == 0)
            != (
                int(r["metadata"]["example_id"][1:]) % 3 == 0
                and r["metadata"]["example_id"] != "n0"
            )
            for r in requests
        ]
        into = tmp_path / "labels.jsonl"
        merge_labels(_filled(requests, labels), into)
        validation = judge_validation(
            judge, into, input_type=int, output_type=str, sealed_splits=()
        )
        result = await validation.run(store=store)
        accuracy = result.metric("accuracy(judge)")
        assert accuracy is not None
        # n3, n6, n9 are labeled against the judge.
        assert accuracy.value == pytest.approx(6 / 9)


def test_annotation_values() -> None:
    assert annotation_value("PASS", None) is True
    assert annotation_value(" no ", 1.0) is False
    assert annotation_value("partial", 0.5) == "partial"
    assert annotation_value(None, 4.0) == 4.0
    assert annotation_value("", None) is None


def _annotation(
    name: str,
    *,
    kind: str = "HUMAN",
    label: str | None = None,
    updated: str,
    **ids: str,
) -> dict[str, Any]:
    return {
        "id": f"a-{updated}",
        "name": name,
        "annotator_kind": kind,
        "result": {"label": label, "score": None, "explanation": "why"},
        "metadata": {},
        "identifier": "",
        "source": "APP",
        "user_id": "VXNlcjox",
        "created_at": updated,
        "updated_at": updated,
        **ids,
    }


def _phoenix(missing_traces: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/arize_phoenix_version":
            return httpx.Response(200, text="20.2.1")
        if path == "/v1/projects/evals":
            return httpx.Response(200, json={"data": {"id": "p", "name": "evals"}})
        if path == "/v1/projects/evals/trace_annotations":
            if missing_traces:
                return httpx.Response(404, json={"detail": "traces not found"})
            assert request.url.params.get_list("include_annotation_names") == ["q"]
            return httpx.Response(
                200,
                json={
                    "data": [
                        _annotation(
                            "q",
                            label="pass",
                            updated="2026-10-01T10:00:00Z",
                            trace_id="t1",
                        ),
                        _annotation(
                            "q",
                            kind="LLM",
                            label="fail",
                            updated="2026-10-01T12:00:00Z",
                            trace_id="t2",
                        ),
                    ],
                    "next_cursor": None,
                },
            )
        if path == "/v1/projects/evals/spans":
            spans = [
                {
                    "name": "s",
                    "context": {"trace_id": t, "span_id": f"s-{t}"},
                    "span_kind": "CHAIN",
                    "start_time": "2026-10-01T09:00:00Z",
                    "end_time": "2026-10-01T09:00:01Z",
                    "status_code": "OK",
                }
                for t in request.url.params.get_list("trace_id")
            ]
            return httpx.Response(200, json={"data": spans, "next_cursor": None})
        if path == "/v1/projects/evals/span_annotations":
            return httpx.Response(
                200,
                json={
                    "data": [
                        # Newer than the trace annotation on t1: it wins.
                        _annotation(
                            "q",
                            label="fail",
                            updated="2026-10-01T11:00:00Z",
                            span_id="s-t1",
                        ),
                        _annotation(
                            "q",
                            label="yes",
                            updated="2026-10-01T09:30:00Z",
                            span_id="s-t2",
                        ),
                    ],
                    "next_cursor": None,
                },
            )
        return httpx.Response(404, json={"detail": f"unexpected {path}"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_human_annotations_newest_per_trace() -> None:
    async with PhoenixClient("http://phoenix.test", transport=_phoenix()) as client:
        result = await human_annotations(client, "evals", ["t1", "t2", "t3"], ["q"])
    found = result.by_trace
    assert found["t1"]["q"].label == "fail"
    assert found["t1"]["q"].span_id == "s-t1"
    # The LLM annotation on t2 is not a human label.
    assert found["t2"]["q"].label == "yes"
    assert "t3" not in found
    assert result.missing_traces == []

    async with PhoenixClient(
        "http://phoenix.test", transport=_phoenix(missing_traces=True)
    ) as client:
        partial = await human_annotations(client, "evals", ["t1"], ["q"])
    assert partial.by_trace["t1"]["q"].label == "fail"


class TestCLI:
    @pytest.mark.asyncio
    async def test_sample_import_show(
        self, tmp_path: Path, store: LocalRunStore, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run = await _run(store)
        root = str(store.root)
        to_label = tmp_path / "to_label.jsonl"
        code = main(
            [
                "--root",
                root,
                "labels",
                "sample",
                run.id,
                "--score",
                "judge",
                "-n",
                "4",
                "--against",
                "check",
                "-o",
                str(to_label),
            ]
        )
        assert code == 0
        sampled = json.loads(capsys.readouterr().out)
        assert sampled["records"] == 4
        rows = [json.loads(line) for line in to_label.read_text().splitlines()]
        assert all(row["reference"] is None for row in rows)
        write_records(_filled(rows, [True, False, True, None]), to_label)
        labels = tmp_path / "labels.jsonl"
        assert (
            main(
                [
                    "labels",
                    "import",
                    str(to_label),
                    "--into",
                    str(labels),
                    "--labeler",
                    "ana",
                ]
            )
            == 0
        )
        imported = json.loads(capsys.readouterr().out)
        assert (imported["added"], imported["unlabeled"]) == (3, 1)
        assert main(["labels", "show", str(labels)]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["labels"] == {"judge": {"true": 2, "false": 1}}
        code = main(
            [
                "--root",
                root,
                "labels",
                "sample",
                run.id,
                "-n",
                "100",
                "--exclude",
                str(to_label),
                "-o",
                str(tmp_path / "next.jsonl"),
            ]
        )
        assert code == 0
        assert json.loads(capsys.readouterr().out)["records"] == 5


class TestLabelingRules:
    @pytest.mark.asyncio
    async def test_labels_for_several_scores_on_one_output(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await _run(store)
        judged = sample_for_labeling(run, 2, score="judge", seed=3)
        checked = sample_for_labeling(run, 2, score="check", seed=3, exclude=[])
        same = {r["id"] for r in judged} & {r["id"] for r in checked}
        into = tmp_path / "labels.jsonl"
        merge_labels(_filled(judged, [True, False]), into, labeler="ana")
        result = merge_labels(_filled(checked, [False, True]), into, labeler="ben")
        assert result.conflicts == []
        for record_id in same:
            labeled = Dataset.load(into)[record_id]
            assert set(labeled.reference) == {"judge", "check"}  # type: ignore[arg-type]
            assert labeled.metadata["labeler"] == {"judge": "ana", "check": "ben"}

    def test_typed_equality_null_labels_and_empty_text(self, tmp_path: Path) -> None:
        into = tmp_path / "labels.jsonl"
        base = {"id": "x", "input": {"output": "o"}, "metadata": {"score": "q"}}
        merge_labels([{**base, "reference": True}], into)
        changed = merge_labels([{**base, "reference": 1}], into, replace=True)
        assert changed.updated == 1
        assert Dataset.load(into)["x"].reference == {"q": 1}
        empty = merge_labels([{**base, "id": "y", "reference": "  "}], into)
        assert (empty.unlabeled, empty.added) == (1, 0)
        # An entry left unlabeled does not block its label later.
        write_records(
            [{**base, "id": "z", "reference": None}, *Dataset.load(into).to_records()],
            into,
        )
        filled = merge_labels([{**base, "id": "z", "reference": False}], into)
        assert filled.updated == 1

    def test_nothing_labeled_writes_nothing(self, tmp_path: Path) -> None:
        into = tmp_path / "labels.jsonl"
        result = merge_labels([{"id": "x", "input": {}, "reference": None}], into)
        assert result.unlabeled == 1
        assert not into.exists()
        with pytest.raises(DatasetError, match=r"\.jsonl"):
            merge_labels([], tmp_path / "labels.yaml")

    @pytest.mark.asyncio
    async def test_verdicts_balance_before_strata_and_unscored_come_last(
        self, store: LocalRunStore
    ) -> None:
        @evaluator(name="judge", annotator="LLM")
        def lopsided(ctx: Ctx) -> bool | None:
            if ctx.input >= 36:
                return None  # not applicable
            # Passes only in topic a (the first ten).
            return ctx.input < 10

        dataset = Dataset(
            [
                Example(id=f"n{i}", input=i, metadata={"topic": "a" if i < 10 else "b"})
                for i in range(40)
            ]
        )
        run = await evaluate(_describe, dataset, [lopsided], store=store)
        records = sample_for_labeling(run, 12, score="judge", strata="topic")
        verdicts = [int(r["metadata"]["example_id"][1:]) < 10 for r in records]
        assert verdicts.count(True) == 6
        everything = sample_for_labeling(run, 100, score="judge")
        tail = [int(r["metadata"]["example_id"][1:]) for r in everything[-3:]]
        assert all(i >= 36 for i in tail)

    @pytest.mark.asyncio
    async def test_unknown_scores_and_judge_runs_are_refused(
        self, tmp_path: Path, store: LocalRunStore
    ) -> None:
        run = await _run(store)
        # A score the run does not have yet (labels for a judge to build):
        # sampled, just not spread over verdicts.
        assert len(sample_for_labeling(run, 3, score="later")) == 3
        with pytest.raises(ValueError, match="against"):
            sample_for_labeling(run, 3, score="judge", against="chekc")

        requests = sample_for_labeling(run, 9, score="judge", test_share=0.0)
        into = tmp_path / "labels.jsonl"
        merge_labels(_filled(requests, [True] * 9), into)
        validation = await judge_validation(
            judge, into, input_type=int, output_type=str, sealed_splits=()
        ).run(store=store)
        with pytest.raises(ValueError, match="judge validation run"):
            sample_for_labeling(validation, 3)

    @pytest.mark.asyncio
    async def test_earlier_splits_are_kept_when_the_share_changes(
        self, store: LocalRunStore
    ) -> None:
        run = await _run(store, repetitions=1)
        first = sample_for_labeling(run, 4, score="judge", test_share=0.0)
        splits = known_splits(first)
        assert set(splits.values()) == {"dev"}
        later = sample_for_labeling(
            run, 100, score="judge", test_share=1.0, splits=splits
        )
        for record in later:
            example_id = record["metadata"]["example_id"]
            expected = "dev" if example_id in splits else "test"
            assert record["splits"] == [expected]


def _paged_phoenix(seen_paths: list[str]) -> httpx.MockTransport:
    spans = [
        {
            "name": "s",
            "context": {"trace_id": "t1", "span_id": f"s{i}"},
            "span_kind": "CHAIN",
            "start_time": "2026-10-01T09:00:00Z",
            "end_time": "2026-10-01T09:00:01Z",
            "status_code": "OK",
        }
        for i in range(250)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.raw_path.decode())
        path = request.url.path
        if path == "/arize_phoenix_version":
            return httpx.Response(200, text="20.2.1")
        if path.endswith("/trace_annotations"):
            return httpx.Response(200, json={"data": [], "next_cursor": None})
        if path.endswith("/spans"):
            start = int(request.url.params.get("cursor") or 0)
            page = spans[start : start + 100]
            more = start + 100 < len(spans)
            return httpx.Response(
                200,
                json={"data": page, "next_cursor": str(start + 100) if more else None},
            )
        if path.endswith("/span_annotations"):
            asked = request.url.params.get_list("span_ids")
            data = (
                [
                    _annotation(
                        "q",
                        label="pass",
                        updated="2026-10-01T10:00:00Z",
                        span_id="s249",
                    )
                ]
                if "s249" in asked
                else []
            )
            return httpx.Response(200, json={"data": data, "next_cursor": None})
        if path.startswith("/v1/projects/"):
            if "missing" in path:
                return httpx.Response(404, json={"detail": "not found"})
            return httpx.Response(200, json={"data": {"id": "p"}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_annotations_on_large_traces_and_odd_project_names() -> None:
    from grasp_agents.evals.phoenix.annotations import (  # noqa: PLC0415
        PhoenixProjectNotFoundError,
    )

    paths: list[str] = []
    transport = _paged_phoenix(paths)
    async with PhoenixClient("http://phoenix.test", transport=transport) as client:
        found = await human_annotations(client, "100% evals", ["t1", "t9"], ["q"])
        assert found.by_trace["t1"]["q"].span_id == "s249"
        assert found.missing_traces == ["t9"]
        with pytest.raises(PhoenixProjectNotFoundError):
            await human_annotations(client, "missing", ["t1"], ["q"])
    project_paths = [p for p in paths if p.startswith("/v1/projects/")]
    assert all(p.startswith("/v1/projects/100%25%20evals") for p in project_paths[:4])


def test_sample_refuses_to_overwrite_and_pull_needs_traces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    filled = tmp_path / "round.jsonl"
    write_records(
        [{"id": "x", "input": {"output": "o"}, "reference": True, "metadata": {}}],
        filled,
    )
    code = main(
        ["labels", "sample", "latest", "-o", str(filled), "--exclude", str(filled)]
    )
    assert code == 2
    assert "also excluded" in capsys.readouterr().err
    code = main(["labels", "sample", "latest", "-o", str(filled)])
    assert code == 2
    assert "--force" in capsys.readouterr().err
    code = main(
        [
            "labels",
            "pull",
            str(filled),
            "--project",
            "p",
            "--into",
            str(tmp_path / "l.jsonl"),
        ]
    )
    assert code == 2
    assert "trace id" in capsys.readouterr().err
