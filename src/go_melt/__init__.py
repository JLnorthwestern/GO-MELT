# src/go_melt/__init__.py
"""GO-MELT public API."""

from importlib.metadata import version as _get_version

__version__ = _get_version("go-melt")

# Public convenience imports (lazy: import only names, not heavy modules)
from .cli import main as run_cli  # very small wrapper, cheap to import

__all__ = ["__version__", "run_cli"]
