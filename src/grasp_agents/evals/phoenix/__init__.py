"""
Arize Phoenix as the shared store for datasets and the UI for runs.

Uses the official ``phoenix.client`` SDK (the ``phoenix`` extra) against
Phoenix 20.0 or later, configured by ``PHOENIX_BASE_URL`` /
``PHOENIX_API_KEY``.
"""

from .client import (
    PhoenixClient,
    PhoenixCompatibilityError,
    PhoenixError,
    PhoenixUnavailableError,
    normalize_base_url,
)
from .sync import (
    NUL_REPLACEMENT,
    DatasetPushError,
    DatasetPushResult,
    StaleDatasetError,
    delete_dataset,
    delete_experiment,
    experiment_url,
    from_phoenix_record,
    phoenix_source,
    pull_dataset,
    push_dataset,
    push_run,
    to_phoenix_record,
)

__all__ = [
    "NUL_REPLACEMENT",
    "DatasetPushError",
    "DatasetPushResult",
    "PhoenixClient",
    "PhoenixCompatibilityError",
    "PhoenixError",
    "PhoenixUnavailableError",
    "StaleDatasetError",
    "delete_dataset",
    "delete_experiment",
    "experiment_url",
    "from_phoenix_record",
    "normalize_base_url",
    "phoenix_source",
    "pull_dataset",
    "push_dataset",
    "push_run",
    "to_phoenix_record",
]
