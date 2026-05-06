"""Tests for ``LiteLLM`` per-model settings.

Pins how settings flow into each Router ``model_list`` entry's
``litellm_params`` after the move from a string-keyed
``model_specific_settings`` dict to inline declarations:

- ``main_model_settings`` is merged into the main entry's ``litellm_params``
  on top of ``{"model": model_name}``.
- A fallback declared as a bare string yields ``{"model": name}`` and nothing
  else (env-var auth path).
- A fallback declared as ``LiteLLMModel(name, settings)`` carries its
  per-model settings inline; they reach only that fallback's
  ``litellm_params``.
- Settings on one model never bleed into another.
- The Router's ``fallbacks`` list contains the resolved names regardless of
  whether each fallback was declared as a plain string or a ``LiteLLMModel``.
- The common ``llm_settings`` (completion-time layer) is unaffected by the
  per-model (routing-time) layer — they live on different code paths.

The Router itself adds defaulted keys (e.g. ``use_in_pass_through``) on top
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

from grasp_agents.litellm.lite_llm import LiteLLM, LiteLLMModel


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


def _router_fallback_names(llm: LiteLLM) -> list[str]:
    """Router stores fallbacks as ``[{main_name: [fb1, fb2, ...]}]`` — extract
    the ordered fallback names so tests can assert on them directly.

    Raises ``AssertionError`` explicitly (not via bare ``assert``) so the
    invariant holds under ``python -O``."""
    fallbacks_cfg: list[dict[str, list[str]]] = llm.router.fallbacks  # type: ignore[assignment]
    if len(fallbacks_cfg) != 1:
        raise AssertionError(
            f"Expected exactly one fallback group, got {len(fallbacks_cfg)}: "
            f"{fallbacks_cfg!r}"
        )
    [(_, names)] = fallbacks_cfg[0].items()
    return list(names)


class TestLiteLLMModelClass(unittest.TestCase):
    def test_default_settings_is_empty(self):
        m = LiteLLMModel("gpt-4o")
        self.assertEqual(m.settings, {})

    def test_settings_carried_on_instance(self):
        m = LiteLLMModel("gpt-4o", {"api_key": "k", "api_base": "b"})
        self.assertEqual(m.name, "gpt-4o")
        self.assertEqual(m.settings, {"api_key": "k", "api_base": "b"})


class TestMainModelSettings(unittest.TestCase):
    def test_default_main_model_settings_is_empty_dict(self):
        llm = _build_llm()
        self.assertEqual(llm.main_model_settings, {})

    def test_default_fallbacks_is_empty_list(self):
        llm = _build_llm()
        self.assertEqual(llm.fallbacks, [])

    def test_main_entry_has_only_model_when_no_settings(self):
        llm = _build_llm()
        params = _params_by_name(llm)["gpt-4o-mini"]
        self.assertEqual(params["model"], "gpt-4o-mini")
        self.assertNotIn("api_key", params)
        self.assertNotIn("api_base", params)

    def test_main_model_settings_flow_into_main_entry(self):
        llm = _build_llm(
            main_model_settings={"api_key": "key-main", "api_base": "https://main"}
        )
        params = _params_by_name(llm)["gpt-4o-mini"]
        self.assertEqual(params["model"], "gpt-4o-mini")
        self.assertEqual(params["api_key"], "key-main")
        self.assertEqual(params["api_base"], "https://main")

    def test_main_settings_do_not_bleed_into_fallback(self):
        llm = _build_llm(
            main_model_settings={"api_key": "key-main"},
            fallbacks=["gpt-4o"],
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["api_key"], "key-main")
        self.assertNotIn("api_key", params["gpt-4o"])

    def test_main_model_settings_can_override_seeded_model(self):
        # Implementation builds ``{"model": name}`` then spreads
        # ``main_model_settings``, so a deliberate ``"model"`` override (e.g.
        # routing to a deployment-qualified id) wins.
        llm = _build_llm(main_model_settings={"model": "azure/my-deployment"})
        params = _params_by_name(llm)["gpt-4o-mini"]
        self.assertEqual(params["model"], "azure/my-deployment")


class TestFallbackDeclarations(unittest.TestCase):
    def test_bare_string_fallback_has_no_extra_settings(self):
        llm = _build_llm(fallbacks=["gpt-4o"])
        params = _params_by_name(llm)["gpt-4o"]
        self.assertEqual(params["model"], "gpt-4o")
        self.assertNotIn("api_key", params)
        self.assertNotIn("api_base", params)

    def test_litellm_model_fallback_carries_its_settings(self):
        llm = _build_llm(
            fallbacks=[LiteLLMModel("gpt-4o", {"api_key": "fb-key"})],
        )
        params = _params_by_name(llm)["gpt-4o"]
        self.assertEqual(params["model"], "gpt-4o")
        self.assertEqual(params["api_key"], "fb-key")

    def test_mixed_fallback_forms_coexist(self):
        llm = _build_llm(
            fallbacks=[
                "gpt-4o",
                LiteLLMModel("claude-sonnet-4-5", {"api_key": "ant-key"}),
                "gpt-3.5-turbo",
            ],
        )
        params = _params_by_name(llm)
        self.assertEqual(
            set(params),
            {"gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5", "gpt-3.5-turbo"},
        )
        self.assertNotIn("api_key", params["gpt-4o"])
        self.assertEqual(params["claude-sonnet-4-5"]["api_key"], "ant-key")
        self.assertNotIn("api_key", params["gpt-3.5-turbo"])

    def test_per_fallback_settings_are_independent(self):
        llm = _build_llm(
            fallbacks=[
                LiteLLMModel("gpt-4o", {"api_key": "k1"}),
                LiteLLMModel(
                    "claude-sonnet-4-5",
                    {"api_key": "k2", "api_base": "https://anthropic"},
                ),
            ],
        )
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o"]["api_key"], "k1")
        self.assertNotIn("api_base", params["gpt-4o"])
        self.assertEqual(params["claude-sonnet-4-5"]["api_key"], "k2")
        self.assertEqual(params["claude-sonnet-4-5"]["api_base"], "https://anthropic")

    def test_fallback_settings_do_not_bleed_into_main(self):
        llm = _build_llm(
            fallbacks=[LiteLLMModel("gpt-4o", {"api_key": "fb-only"})],
        )
        main_params = _params_by_name(llm)["gpt-4o-mini"]
        self.assertNotIn("api_key", main_params)


class TestRouterFallbackList(unittest.TestCase):
    def test_router_fallback_names_from_bare_strings(self):
        llm = _build_llm(fallbacks=["gpt-4o", "gpt-3.5-turbo"])
        self.assertEqual(_router_fallback_names(llm), ["gpt-4o", "gpt-3.5-turbo"])

    def test_router_fallback_names_from_litellm_models(self):
        llm = _build_llm(
            fallbacks=[
                LiteLLMModel("gpt-4o", {"api_key": "k"}),
                LiteLLMModel("claude-sonnet-4-5"),
            ],
        )
        self.assertEqual(
            _router_fallback_names(llm), ["gpt-4o", "claude-sonnet-4-5"]
        )

    def test_router_fallback_names_preserve_order_in_mixed_input(self):
        llm = _build_llm(
            fallbacks=[
                "gpt-3.5-turbo",
                LiteLLMModel("claude-sonnet-4-5", {"api_key": "k"}),
                "gpt-4o",
            ],
        )
        self.assertEqual(
            _router_fallback_names(llm),
            ["gpt-3.5-turbo", "claude-sonnet-4-5", "gpt-4o"],
        )


class TestSettingsLayeringIsolation(unittest.TestCase):
    """Pins that the per-model (routing-time) layer doesn't disturb the
    common ``llm_settings`` (completion-time) layer — they live on different
    code paths and shouldn't cross-contaminate."""

    def test_llm_settings_does_not_leak_into_per_model_litellm_params(self):
        llm = _build_llm(
            llm_settings={"temperature": 0.42},
            main_model_settings={"api_key": "key-main"},
            fallbacks=[LiteLLMModel("gpt-4o", {"api_key": "fb-k"})],
        )
        # llm_settings stays on the LLM instance for the completion path.
        self.assertIsNotNone(llm.llm_settings)
        # Cast for the type checker — the runtime check above is what guards
        # the test, and survives ``python -O`` (unlike a bare ``assert``).
        llm_settings = cast("dict[str, Any]", llm.llm_settings)
        self.assertEqual(llm_settings.get("temperature"), 0.42)
        # Per-model layer still works.
        params = _params_by_name(llm)
        self.assertEqual(params["gpt-4o-mini"]["api_key"], "key-main")
        self.assertEqual(params["gpt-4o"]["api_key"], "fb-k")
        # llm_settings does NOT leak into Router-level litellm_params.
        for entry in params.values():
            self.assertNotIn("temperature", entry)


if __name__ == "__main__":
    unittest.main()
