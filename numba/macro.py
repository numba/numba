"""
Macro handling.

Macros are expanded on block-by-block
"""

# Expose the Macro object from the corresponding IR rewrite pass
from numba.core.rewrites.macros import Macro
