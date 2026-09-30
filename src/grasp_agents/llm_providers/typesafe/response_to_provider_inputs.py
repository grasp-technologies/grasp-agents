"""Convert input items → Jev's state and the framing shared by every question."""

from collections.abc import Sequence

from grasp_agents.types.content import InputText
from grasp_agents.types.items import InputItem, InputMessageItem

_FRAMING_ROLES = frozenset({"system", "developer"})


def items_to_state(input: Sequence[InputItem]) -> tuple[str, str | None]:  # ruff: ignore[builtin-argument-shadowing]
    """
    Return ``(state, framing)``.

    Jev judges one state per request, so the input must hold exactly one user
    message; system and developer messages become framing, never state.
    """
    user_texts: list[str] = []
    framing_texts: list[str] = []
    for item in input:
        if not isinstance(item, InputMessageItem):
            raise TypeError(
                f"TypeSafeLLM cannot send a {type(item).__name__}: Jev judges one"
                " user message, not a conversation"
            )
        if any(not isinstance(part, InputText) for part in item.content):
            raise ValueError("TypeSafeLLM sends text only, not images or files")
        text = "\n\n".join(item.texts)
        if item.role in _FRAMING_ROLES:
            framing_texts.append(text)
        else:
            user_texts.append(text)

    if len(user_texts) != 1:
        raise ValueError(
            f"TypeSafeLLM needs exactly one user message, got {len(user_texts)}"
        )
    return user_texts[0], "\n\n".join(framing_texts) or None
