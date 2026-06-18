"""Tests for ``LiteLLM`` per-model settings.

Pins the contract of the two mutually-exclusive ways to configure settings:

- ``llm_settings``: a single settings dict applied to all models (common,
  completion-time layer — flows through the Router as call kwargs).
- ``llm_group_settings``: a per-model settings dict ``{model_name: settings}``
  baked into each Router ``model_list`` entry's ``litellm_params``. When
  provided, it must contain *exactly* the routed models (``model_name`` plus
  every entry of ``fallbacks``) — no missing keys, no extras. ``{}`` is the
  explicit way to say "this model is routed but needs no specific settings."

Supplying both raises ``ValueError``. Supplying neither leaves every Router
entry with only ``{"model": name}``.

``fallbacks`` on ``LiteLLM`` is a plain ``list[str]`` of model names; we
internally wrap it into litellm Router's native ``[{main: [fb1, fb2]}]``
shape at the boundary.

The Router itself adds defaulted keys (``use_in_pass_through``, etc.) on top
of what we pass, so assertions check that *our* keys are present with the
right values rather than equality on the whole ``litellm_params`` dict.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from grasp_agents.litellm.lite_llm import LiteLLM


def _build_llm(**overrides: Any) -> LiteLLM:
    kwargs: dict[str, Any] = {
        "model_name": "gpt-4o-mini",
        "mock_response": "hi",
    }
    kwargs.update(overrides)
    return LiteLLM(**kwargs)


def _params_by_name(llm: LiteLLM) -> dict[str, dict[str, Any]]:
    model_list: list[dict[str, Any]] = llm.router.model_list  # type: ignore[assignment]
    return {entry["model_name"]: entry["litellm_params"] for entry in model_list}


class TestPlainConstruction(unittest.TestCase):
    """No settings, no fallbacks: a single Router entry for the main model
    with only ``{"model": name}`` in its ``litellm_params``."""

    def test_default_llm_settings_is_none(self):
        llm = _build_llm()
        self.assertIsNone(llm.llm_settings)

    def test_default_llm_group_settings_is_none(self):
        llm = _build_llm()
        self.assertIsNone(llm.llm_group_settings)

    def test_default_fallbacks_is_empty_list(self):
        llm = _build_llm()
        self.assertEqual(llm.fallbacks, [])

    def test_main_model_is_only_router_entry(self):
        llm = _build_llm()
        params = _params_by_name(llm)
        self.assertEqual(set(params), {"gpt-4o-mini"})
        self.assertEqual(params["gpt-4o-mini"]["model"], "gpt-4o-mini")
        self.assertNotIn("api_key", params["gpt-4o-mini"])


class TestLlmSettingsCommon(unittest.TestCase):
    """``llm_settings`` is the common (completion-time) layer. It does not
    appear in per-model ``litellm_params`` — it flows through the call path
    instead. The Router's ``model_list`` should contain only ``{"model": ...}``
    for each entry."""

    def test_llm_settings_stored_on_instance(self):
        llm = _build_llm(llm_settings={"temperature": 0.42})
        self.assertIsNotNone(llm.llm_settings)
        # cast for the type checker; runtime guard is the assertIsNotNone above.
        llm_settings = cast("dict[str, Any]", llm.llm_settings)
        self.assertEqual(llm_settings.get("temperature"), 0.42)

    def test_llm_settings_does_not_leak_into_litellm_params(self):
        llm = _build_llm(
            llm_settings={"temperature": 0.42, "max_completion_tokens": 100},
            fallbacks=["gpt-4o"],
        )
        params = _params_by_name(llm)
        for entry in params.values():
            self.assertNotIn("temperature", entry)
            self.assertNotIn("max_completion_tokens", entry)

    def test_llm_settings_with_fallbacks_includes_all_routed_models(self):
        llm = _build_llm(
            llm_settings={"temperature": 0.5},
            fallbacks=["gpt-4o", "claude-sonnet-4-5"],
        )
        params = _params_by_name(llm)
        self.assertEqual(
            set(params), {"gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5"}
        )


class TestLlmGroupSettingsPerModel(unittest.TestCase):
    """``llm_group_settings`` carries per-model settings into each entry's
    ``litellm_params``. Settings on one model never bleed into another."""

    def test_per_model_settings_reach_each_entry(self):
        llm = _build_llm(
            fallbacks=["gpt-4o"],
            llm_group_settings={
                "gpt-4o-mini": {"api_key": "K-main", "api_base": "https://main"},
                "gpt-4o":      {"api_key": "K-fb"},
            },
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["api_key"], "K-main")
        self.assertEqual(params["gpt-4o-mini"]["api_base"], "https://main")
        self.assertEqual(params["gpt-4o"]["api_key"], "K-fb")
        # api_base set only on main, not on fallback.
        self.assertNotIn("api_base", params["gpt-4o"])

    def test_empty_dict_means_no_settings_for_that_model(self):
        # An empty dict is the explicit way to declare a routed model with
        # no specific settings. The entry exists in model_list but with only
        # ``{"model": name}``.
        llm = _build_llm(
            fallbacks=["gpt-4o"],
            llm_group_settings={
                "gpt-4o-mini": {"api_key": "K"},
                "gpt-4o":      {},
            },
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["api_key"], "K")
        self.assertEqual(params["gpt-4o"]["model"], "gpt-4o")
        self.assertNotIn("api_key", params["gpt-4o"])

    def test_per_model_settings_do_not_bleed_between_entries(self):
        llm = _build_llm(
            fallbacks=["gpt-4o", "claude-sonnet-4-5"],
            llm_group_settings={
                "gpt-4o-mini":      {"api_key": "K1"},
                "gpt-4o":           {"api_key": "K2", "api_base": "https://b"},
                "claude-sonnet-4-5": {},
            },
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["api_key"], "K1")
        self.assertNotIn("api_base", params["gpt-4o-mini"])
        self.assertEqual(params["gpt-4o"]["api_key"], "K2")
        self.assertEqual(params["gpt-4o"]["api_base"], "https://b")
        self.assertNotIn("api_key", params["claude-sonnet-4-5"])

    def test_settings_can_override_seeded_model(self):
        # The ``"model"`` key is seeded as the entry name and then any
        # per-model settings are spread on top, so a deliberate ``"model"``
        # override (e.g. routing to a deployment-qualified id) wins.
        llm = _build_llm(
            llm_group_settings={
                "gpt-4o-mini": {"model": "azure/my-deployment"},
            },
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["model"], "azure/my-deployment")


class TestStrictValidation(unittest.TestCase):
    """Construction must reject configurations that would silently misbehave
    or leave the Router with missing entries."""

    def test_mutex_llm_settings_and_llm_group_settings(self):
        with self.assertRaises(ValueError) as ctx:
            _build_llm(
                llm_settings={"temperature": 0.5},
                llm_group_settings={"gpt-4o-mini": {}},
            )
        self.assertIn("llm_settings", str(ctx.exception))
        self.assertIn("llm_group_settings", str(ctx.exception))

    def test_missing_main_model_in_group_settings(self):
        with self.assertRaises(ValueError) as ctx:
            _build_llm(
                fallbacks=["gpt-4o"],
                llm_group_settings={"gpt-4o": {}},
            )
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("gpt-4o-mini", str(ctx.exception))

    def test_missing_fallback_target_in_group_settings(self):
        with self.assertRaises(ValueError) as ctx:
            _build_llm(
                fallbacks=["gpt-4o"],
                llm_group_settings={"gpt-4o-mini": {}},
            )
        self.assertIn("Missing", str(ctx.exception))
        self.assertIn("gpt-4o", str(ctx.exception))

    def test_extra_key_in_group_settings(self):
        with self.assertRaises(ValueError) as ctx:
            _build_llm(
                llm_group_settings={
                    "gpt-4o-mini": {},
                    "stray-model": {"api_key": "K"},
                },
            )
        self.assertIn("Unexpected", str(ctx.exception))
        self.assertIn("stray-model", str(ctx.exception))

    def test_typo_reported_as_both_missing_and_extra(self):
        # "gpt-4-mini" looks like a typo for "gpt-4o-mini". The strict check
        # surfaces the mismatch from both sides at once so the user can see
        # exactly what went wrong.
        with self.assertRaises(ValueError) as ctx:
            _build_llm(
                llm_group_settings={"gpt-4-mini": {"api_key": "K"}},
            )
        msg = str(ctx.exception)
        self.assertIn("Missing", msg)
        self.assertIn("gpt-4o-mini", msg)
        self.assertIn("Unexpected", msg)
        self.assertIn("gpt-4-mini", msg)


class TestFallbacksWithoutGroupSettings(unittest.TestCase):
    """When ``llm_group_settings`` is not provided, the else branch builds
    ``model_list`` from the union of the main model and every fallback
    target. Each entry gets only ``{"model": name}`` (env-var auth path)."""

    def test_model_list_includes_main_and_all_fallback_targets(self):
        llm = _build_llm(
            fallbacks=["gpt-4o", "claude-sonnet-4-5"],
        )
        params = _params_by_name(llm)
        self.assertEqual(
            set(params), {"gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5"}
        )

    def test_each_fallback_entry_has_no_extra_settings(self):
        llm = _build_llm(fallbacks=["gpt-4o"])
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o"]["model"], "gpt-4o")
        self.assertNotIn("api_key", params["gpt-4o"])
        self.assertNotIn("api_base", params["gpt-4o"])

    def test_main_appearing_in_fallbacks_is_deduped(self):
        # Pathological but possible: main is also listed as a fallback. The
        # else branch builds model_list from a set, so the entry appears
        # exactly once.
        llm = _build_llm(fallbacks=["gpt-4o-mini", "gpt-4o"])
        model_list: list[dict[str, Any]] = llm.router.model_list  # type: ignore[assignment]
        names = [e["model_name"] for e in model_list]
        self.assertEqual(names.count("gpt-4o-mini"), 1)
        self.assertEqual(set(names), {"gpt-4o-mini", "gpt-4o"})

    def test_duplicate_fallback_names_dedupe_in_model_list(self):
        # A duplicate in fallbacks shouldn't produce a duplicate Router entry
        # (set-union dedupes). The Router's fallback list is passed through
        # as-is, so duplicates there are litellm's concern.
        llm = _build_llm(fallbacks=["gpt-4o", "gpt-4o"])
        params = _params_by_name(llm)
        self.assertEqual(set(params), {"gpt-4o-mini", "gpt-4o"})


class TestRouterFallbacksWrapping(unittest.TestCase):
    """The public ``fallbacks: list[str]`` is wrapped into litellm Router's
    native ``[{main_name: [fb1, fb2]}]`` shape at the Router boundary."""

    def test_non_empty_fallbacks_wrapped_for_router(self):
        llm = _build_llm(fallbacks=["gpt-4o", "claude-sonnet-4-5"])
        router_fallbacks: list[dict[str, list[str]]] = llm.router.fallbacks  # type: ignore[assignment]
        self.assertEqual(
            router_fallbacks,
            [{"gpt-4o-mini": ["gpt-4o", "claude-sonnet-4-5"]}],
        )

    def test_empty_fallbacks_still_wrapped(self):
        llm = _build_llm()
        router_fallbacks: list[dict[str, list[str]]] = llm.router.fallbacks  # type: ignore[assignment]
        self.assertEqual(router_fallbacks, [{"gpt-4o-mini": []}])


if __name__ == "__main__":
    unittest.main()
