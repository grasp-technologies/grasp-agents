__all__ = ["format_error_chain"]


def format_error_chain(error: BaseException) -> str:
    """
    Format an exception together with its cause chain, which ``str()`` alone
    drops — a wrapper error's string would otherwise hide the root cause.
    """
    parts: list[str] = []
    seen: set[int] = set()
    err: BaseException | None = error
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        text = str(err)
        parts.append(f"{type(err).__name__}: {text}" if text else type(err).__name__)
        if err.__cause__ is not None:
            err = err.__cause__
        elif not err.__suppress_context__:
            err = err.__context__
        else:
            err = None
    return "\nCaused by: ".join(parts)
