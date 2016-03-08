"""
A subpackage hosting Numba IR rewrite passes.
"""

from .registry import register_rewrite, rewrite_registry, Rewrite

# Register various built-in rewrite passes
from . import static_getitem, static_raise, macros
