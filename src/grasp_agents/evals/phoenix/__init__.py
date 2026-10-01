"""
Arize Phoenix as the shared store for datasets and the UI for runs.

Talks to Phoenix's REST API directly (``PHOENIX_BASE_URL`` /
``PHOENIX_API_KEY``). Works with server 12.18+; full dataset sync and
client-supplied example ids need server 15+.
"""

from .client import PhoenixClient, PhoenixConflictError, PhoenixError
from .sync import (
    PhoenixCompatibilityError,
    StaleDatasetError,
    experiment_url,
    from_phoenix_record,
    pull_dataset,
    push_dataset,
    push_run,
    to_phoenix_record,
)

__all__ = [
    "PhoenixClient",
    "PhoenixCompatibilityError",
    "PhoenixConflictError",
    "PhoenixError",
    "StaleDatasetError",
    "experiment_url",
    "from_phoenix_record",
    "pull_dataset",
    "push_dataset",
    "push_run",
    "to_phoenix_record",
]
