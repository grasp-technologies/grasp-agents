from typing import TypedDict


class BedrockClientConfig(TypedDict, total=False):
    """
    Client args for ``platform="bedrock"`` / ``platform="bedrock_mantle"``, on
    ``AnthropicLLM`` and the OpenAI classes alike.

    Unset values fall back to the standard AWS credential/region chain (env
    vars, shared config/credentials files, SSO, IMDS). ``api_key`` is a Bedrock
    API key (a bearer token, AWS_BEARER_TOKEN_BEDROCK when unset), mutually
    exclusive with the ``aws_*`` credential fields.
    """

    aws_region: str
    aws_profile: str
    aws_access_key: str
    aws_secret_key: str
    aws_session_token: str
    api_key: str
    base_url: str
