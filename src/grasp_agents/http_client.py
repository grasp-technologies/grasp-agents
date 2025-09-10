import httpx
from pydantic import BaseModel, NonNegativeFloat, PositiveInt


class AsyncHTTPClientParams(BaseModel):
    timeout: NonNegativeFloat = 10
    max_connections: PositiveInt = 2000
    max_keepalive_connections: PositiveInt = 500
    keepalive_expiry: float | None = 5
    # proxy: str | None = None
    # follow_redirects: bool = False
    # trust_env: bool = True
    # auth: AuthTypes | None = (None,)
    # params: QueryParamTypes | None = (None,)
    # headers: HeaderTypes | None = (None,)
    # cookies: CookieTypes | None = (None,)


def create_simple_async_httpx_client(
    client_params: AsyncHTTPClientParams,
) -> httpx.AsyncClient:
    extra_params = client_params.model_dump(
        exclude={
            "timeout",
            "max_connections",
            "max_keepalive_connections",
            "keepalive_expiry",
        }
    )
    return httpx.AsyncClient(
        timeout=httpx.Timeout(client_params.timeout),
        limits=httpx.Limits(
            max_connections=client_params.max_connections,
            max_keepalive_connections=client_params.max_keepalive_connections,
            keepalive_expiry=client_params.keepalive_expiry,
        ),
        **extra_params,
    )
