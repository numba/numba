from __future__ import print_function, division, absolute_import

from .typing.typeof import typeof
from .parfor import prange

def set_user_pipeline_func(func):
    from .compiler import set_user_pipeline_func
    set_user_pipeline_func(func)

__all__ = ['typeof','prange','set_user_pipeline_func']
