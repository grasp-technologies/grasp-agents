"""Tests for ``LiteLLM.model_specific_settings``.

Pins how per-model overrides supplied via ``model_specific_settings`` are
merged into each Router entry's ``litellm_params``:

- The main model receives its overrides on top of ``{"model": <name>}``.
- Each fallback receives its own overrides independently.
- A model entry that matches neither the main model nor a declared fallback is
  silently ignored (it has no Router entry to attach to).
- When ``model_specific_settings`` is empty, no per-model overrides leak in.

The Router itself adds defaulted keys (e.g. ``use_in_pass_through``,
``merge_reasoning_content_in_choices``) on top of what we pass, so assertions
check that our keys are present with the right values rather than equality on
the whole ``litellm_params`` dict.
"""

import sys
import unittest
from pathlib import Path
from typing import Any

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


class TestLiteLLMModelSpecificSettings(unittest.TestCase):
    def test_default_field_is_empty_dict(self):
        llm = _build_llm()
        self.assertEqual(llm.model_specific_settings, {})

    def test_no_overrides_when_settings_empty(self):
        llm = _build_llm(fallbacks=["gpt-4o"])
        params = _params_by_name(llm)

        self.assertEqual(set(params), {"gpt-4o-mini", "gpt-4o"})
        for name, entry in params.items():
            self.assertEqual(entry["model"], name)
            self.assertNotIn("api_key", entry)
            self.assertNotIn("api_base", entry)

    def test_overrides_applied_to_main_model(self):
        llm = _build_llm(
            model_specific_settings={
                "gpt-4o-mini": {"api_key": "key-main", "api_base": "https://main"}
            },
        )
        main_params = _params_by_name(llm)["gpt-4o-mini"]

        self.assertEqual(main_params["model"], "gpt-4o-mini")
        self.assertEqual(main_params["api_key"], "key-main")
        self.assertEqual(main_params["api_base"], "https://main")

    def test_overrides_applied_to_fallback_only(self):
        llm = _build_llm(
            fallbacks=["gpt-4o"],
            model_specific_settings={"gpt-4o": {"api_key": "key-fb"}},
        )
        params = _params_by_name(llm)

        self.assertNotIn("api_key", params["gpt-4o-mini"])
        self.assertEqual(params["gpt-4o"]["api_key"], "key-fb")
        self.assertEqual(params["gpt-4o"]["model"], "gpt-4o")

    def test_overrides_are_independent_per_model(self):
        llm = _build_llm(
            fallbacks=["gpt-4o", "gpt-3.5-turbo"],
            model_specific_settings={
                "gpt-4o-mini": {"api_key": "main-key"},
                "gpt-3.5-turbo": {"api_key": "legacy-key", "api_base": "https://legacy"},
            },
        )
        params = _params_by_name(llm)

        self.assertEqual(params["gpt-4o-mini"]["api_key"], "main-key")
        # Fallback without an override entry must not pick up another model's keys.
        self.assertNotIn("api_key", params["gpt-4o"])
        self.assertEqual(params["gpt-3.5-turbo"]["api_key"], "legacy-key")
        self.assertEqual(params["gpt-3.5-turbo"]["api_base"], "https://legacy")

    def test_settings_for_unknown_model_are_ignored(self):
        llm = _build_llm(
            fallbacks=["gpt-4o"],
            model_specific_settings={"some-other-model": {"api_key": "stray"}},
        )
        params = _params_by_name(llm)

        # Stray entry produces no Router model and never bleeds into the others.
        self.assertEqual(set(params), {"gpt-4o-mini", "gpt-4o"})
        for entry in params.values():
            self.assertNotIn("api_key", entry)

    def test_override_can_replace_model_field(self):
        # The implementation seeds ``{"model": <name>}`` first and then applies
        # ``model_specific_settings[name]`` via ``dict.update``, so a deliberate
        # ``"model"`` override (e.g. routing to a deployment-qualified id) wins.
        llm = _build_llm(
            model_specific_settings={
                "gpt-4o-mini": {"model": "azure/my-deployment"}
            },
        )
        main_params = _params_by_name(llm)["gpt-4o-mini"]

        self.assertEqual(main_params["model"], "azure/my-deployment")


if __name__ == "__main__":
    unittest.main()
