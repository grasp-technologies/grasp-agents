import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def _e2b_trusts_ssl_cert_file() -> Iterator[None]:
    """
    Make the E2B SDK verify TLS against ``$SSL_CERT_FILE``.

    E2B's transport trusts only the OS certificate store, so behind a
    TLS-intercepting proxy whose CA is only in ``$SSL_CERT_FILE`` (e.g. a
    command sandbox) every E2B call fails certificate verification.
    """
    ca_file = os.environ.get("SSL_CERT_FILE")
    if not ca_file:
        yield
        return
    try:
        from e2b.api import client_async, client_sync
        from pyqwest import HTTPTransport, SyncHTTPTransport
    except ImportError:
        yield
        return

    ca = Path(ca_file).read_bytes()

    def trusting(transport_cls: Callable[..., Any]) -> Callable[..., Any]:
        def make(*args: Any, **kwargs: Any) -> Any:
            kwargs["tls_include_system_certs"] = False
            kwargs["tls_ca_cert"] = ca
            return transport_cls(*args, **kwargs)

        return make

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(client_async, "HTTPTransport", trusting(HTTPTransport))
        mp.setattr(client_sync, "SyncHTTPTransport", trusting(SyncHTTPTransport))
        yield
