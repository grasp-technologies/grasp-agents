from logging import getLogger
from typing import Any

logger = getLogger(__name__)


def is_method_overridden(
    method_name: str, self: object, base_cls: type[Any] | None = None
) -> bool:
    """
    Check if a method is overridden in a subclass compared to a base class
    or if it is defined directly in the instance's __dict__ (e.g. via a decorator).
    """
    child_cls = type(self)
    if not hasattr(child_cls, method_name):
        raise AttributeError(f"{child_cls} has no method named {method_name}")

    set_via_decorator = method_name in self.__dict__

    overriden_in_child = False
    if base_cls is not None:
        overriden_in_child = hasattr(base_cls, method_name) and getattr(
            child_cls, method_name
        ) is not getattr(base_cls, method_name)

    return set_via_decorator or overriden_in_child
