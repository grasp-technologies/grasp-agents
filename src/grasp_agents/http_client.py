from typing import Any

import httpx
from pydantic import BaseModel, NonNegativeFloat, PositiveInt


class AsyncHTTPClientParams(BaseModel):
    timeout: NonNegativeFloat = 10
    max_connections: PositiveInt = 2000
    max_keepalive_connections: PositiveInt = 500
    keepalive_expiry: float | None = 5


def create_simple_async_httpx_client(
    client_params: AsyncHTTPClientParams | dict[str, Any],
) -> httpx.AsyncClient:
    if isinstance(client_params, dict):
        client_params = AsyncHTTPClientParams(**client_params)
    return httpx.AsyncClient(
        timeout=httpx.Timeout(client_params.timeout),
        limits=httpx.Limits(
            max_connections=client_params.max_connections,
            max_keepalive_connections=client_params.max_keepalive_connections,
            keepalive_expiry=client_params.keepalive_expiry,
        ),
    )
