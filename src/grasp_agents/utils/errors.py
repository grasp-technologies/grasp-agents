__all__ = ["format_error_chain", "root_cause"]


def _next_in_chain(err: BaseException) -> BaseException | None:
    if err.__cause__ is not None:
        return err.__cause__
    if not err.__suppress_context__:
        return err.__context__
    return None


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
        err = _next_in_chain(err)
    return "\nCaused by: ".join(parts)


def root_cause(error: BaseException) -> BaseException:
    """
    Walk an exception's cause chain to its deepest cause.

    Framework wrappers (e.g. a processor's retry ``ProcRunError``) sit at the
    top of the chain; the root is usually the error a caller actually cares to
    branch on.
    """
    seen: set[int] = set()
    err = error
    while id(err) not in seen:
        seen.add(id(err))
        nxt = _next_in_chain(err)
        if nxt is None:
            return err
        err = nxt
    return err
