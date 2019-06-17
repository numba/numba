"""
Macro handling.

Macros are expanded on block-by-block
"""
from __future__ import absolute_import, print_function, division

# Expose the Macro object from the corresponding IR rewrite pass
from .rewrites.macros import Macro
