"""Strategy registry.

Decorate a :class:`~framework.strategies.base.Strategy` subclass with
``@register`` and it becomes addressable by its ``name`` from any config
file. Both the committed ``examples`` package and the gitignored ``private``
package are imported below, so dropping a new file into either is all the
registration ceremony there is.
"""

from __future__ import annotations

from framework.strategies.base import Strategy, StrategyParams

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator adding a Strategy subclass to the registry by its ``name``."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} must define a class-level `name`")
    if cls.name in _REGISTRY:
        raise ValueError(f"Duplicate strategy name {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> type[Strategy]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {name!r}. Available: {available_strategies()}"
        ) from None


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


# Import strategy packages so their @register decorators run.
from framework.strategies import examples as _examples  # noqa: E402,F401
from framework.strategies import private as _private  # noqa: E402,F401

__all__ = [
    "Strategy",
    "StrategyParams",
    "register",
    "get_strategy",
    "available_strategies",
]
