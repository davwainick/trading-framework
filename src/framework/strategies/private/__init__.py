"""Private strategies live here. Every module except this file and README.md
is gitignored — see README.md in this directory for the convention.

Modules in this package are auto-imported so their @register decorators run.
"""

import importlib
import pkgutil

for _mod in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_mod.name}")
