import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import unittest

from grasp_agents.telemetry.exporters import FilteringExporter


class _DummyInnerExporter:
    def __init__(self) -> None:
        self.exported_spans: list[Any] | None = None

    def export(self, spans: list[Any]) -> str:
        self.exported_spans = list(spans)
        return "OK"

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _DummySpan:
    def __init__(self, attributes: dict[str, Any], name: str = "dummy") -> None:
        self.attributes = attributes
        self.name = name


class TestFilteringExporter(unittest.TestCase):
    def test_filter_based_on_attrs_blocks_http_metadata_domain(self):
        inner = _DummyInnerExporter()
        exporter = FilteringExporter(
            inner=inner, attribute_filter={"http": {"metadata.google.internal"}}
        )

        attrs = {
            "http": {
                "url": (
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/"
                    "core-service-account@grasp-core-prod.iam.gserviceaccount.com/?recursive=true"
                ),
                "method": "GET",
                "status_code": 200,
            }
        }

        # Act
        allowed = exporter._filter_based_on_attrs(attrs)

        self.assertFalse(allowed)

    def test_export_drops_span_with_metadata_http_and_keeps_others(self):
        inner = _DummyInnerExporter()
        exporter = FilteringExporter(
            inner=inner, attribute_filter={"http": {"metadata.google.internal"}}
        )

        metadata_attrs = {
            "http": {
                "url": (
                    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/"
                    "core-service-account@grasp-core-prod.iam.gserviceaccount.com/?recursive=true"
                ),
                "method": "GET",
                "status_code": 200,
            }
        }
        safe_attrs = {
            "http": {
                "url": "http://example.com",
                "method": "GET",
                "status_code": 200,
            }
        }

        spans = [
            _DummySpan(metadata_attrs, name="metadata"),
            _DummySpan(safe_attrs, name="safe"),
        ]

        exporter.export(spans)

        self.assertIsNotNone(inner.exported_spans)
        self.assertEqual(len(inner.exported_spans), 1)
        self.assertEqual(getattr(inner.exported_spans[0], "name", ""), "safe")
