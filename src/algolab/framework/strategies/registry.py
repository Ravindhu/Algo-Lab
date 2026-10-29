"""Strategy plugin registry: add a strategy by writing one file, no other edits.

How to plug in a new strategy (plug and play):

1. Create a file in this package, e.g. ``strategies/my_strategy.py``::

       from algolab.framework.strategies.base import Strategy

       class MyStrategy(Strategy):
           name = "my_strategy"           # becomes the "type" in config/framework.json

           def signal(self, df):
               # pure rule on CLOSED bars; return {"side": "BUY"|"SELL", ...} or None
               ...

       # make it discoverable:
       STRATEGY = MyStrategy              # the registry looks for this name

2. Add it to config/framework.json::

       {"type": "my_strategy", "params": {"timeframe": "M15", ...}, "enabled": true}

Nothing else changes: the runner loads it by name, the Bot handles gates/sizing/orders,
and the position manager handles exits.

Registry contract:
  - Each strategy module exposes ``STRATEGY`` (the class, or a list of classes).
  - ``register()`` maps each class's ``name`` to the class.
  - ``load(name)`` imports the module lazily on first use, so an import error in one
    strategy file cannot break the others until that strategy is actually requested.
  - ``enabled(entry)`` honours the per-strategy ``"enabled": false`` flag in config
    (default true), so a strategy can ship disabled.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

log = logging.getLogger("algolab.strategies")

# module name inside this package -> attribute holding the class (or a list of them)
_PLUGINS: dict[str, str] = {}


def register(module_name: str, attr: str = "STRATEGY") -> None:
    """Announce a plugin module. The class is imported lazily by load()."""
    _PLUGINS[module_name] = attr


# Built-in plugins. Add one line here OR call register() from the strategy file itself;
# both work. Keeping the list here makes the full menu visible at a glance.
for _mod in ("scalp", "engulfing_strategy", "ocl", "ocl_sell_fail", "ocl_inverse", "breakout_example"):
    register(_mod)


def available() -> list[str]:
    """Strategy names that are registered (not necessarily imported)."""
    return list(_PLUGINS)


def _extract(module: Any, attr: str) -> list[type]:
    raw = getattr(module, attr, None)
    if raw is None:
        raise AttributeError(f"{module.__name__} has no {attr!r} attribute")
    classes = raw if isinstance(raw, (list, tuple)) else [raw]
    out = []
    for cls in classes:
        if not (isinstance(cls, type) and getattr(cls, "name", None)):
            raise TypeError(f"{module.__name__}: plugin entries must be classes with a 'name'")
        out.append(cls)
    return out


_PACKAGE = __name__.rsplit(".", 1)[0]          # algolab.framework.strategies


def _all_classes(module_name: str, attr: str) -> list[type]:
    module = importlib.import_module(f"{_PACKAGE}.{module_name}")
    return _extract(module, attr)


def load(name: str) -> type:
    """Strategy class by its config 'type' (the class's `name`). Imported lazily."""
    for module_name, attr in _PLUGINS.items():
        try:
            classes = _all_classes(module_name, attr)
        except Exception as e:                      # a broken plugin must not break the rest
            log.error("strategy plugin %s failed to load: %s", module_name, e)
            continue
        for cls in classes:
            if cls.name == name:
                return cls
    raise KeyError(f"unknown strategy type {name!r}; available: "
                   f"{sorted({c.name for mn, a in _PLUGINS.items() for c in _safe(mn, a)})}")


def _safe(module_name: str, attr: str) -> list[type]:
    try:
        return _all_classes(module_name, attr)
    except Exception:
        return []


def enabled(entry: dict) -> bool:
    """Per-strategy on/off switch from config/framework.json. Default is on."""
    return bool(entry.get("enabled", True))
