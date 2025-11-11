# Expose all immediate submodules as attributes, so r.<tab> shows them.
import importlib, pkgutil as _pkgutil

__all__ = []
for _finder, _name, _ispkg in _pkgutil.iter_modules(__path__):
    globals()[_name] = importlib.import_module(f".{_name}", __name__)
    __all__.append(_name)

# clean up
del _finder, _name, _ispkg, importlib, _pkgutil

