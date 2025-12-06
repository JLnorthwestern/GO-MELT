# src/go_melt/core/__init__.py
"""Core numerical routines."""

from .go_melt import go_melt as run_simulation  # entrypoint for CLI

__all__ = ["run_simulation"]
