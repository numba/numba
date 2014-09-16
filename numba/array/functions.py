from __future__ import print_function, absolute_import
from .array import Array
import numpy as np
from .nodes import WhereOperation

python_max = max # max gets replaced by the array max method down below

import sys as sys
api = sys.modules[__name__]


array_creation_methods = ['array', 'empty', 'eye', 'identity', 'ones', 'zeros',
    'arange', 'linspace', 'logspace', 'meshgrid']

for method in array_creation_methods:
    def create_method_template(method):
        def method_template(*args, **kwargs):
            return Array(data=getattr(np, method)(*args, **kwargs))
        return method_template
    setattr(api, method, create_method_template(method))


# All of the following methods force the deferred array to be evaluated
array_reduce_methods = ['amin', 'amax', 'nanmin', 'nanmax', 'ptp', 'max',
    'percentile', 'median', 'average', 'mean', 'std', 'var', 'nanmean',
    'nanstd', 'nanvar', 'corrcoef', 'correlate', 'cov', 'histogram',
    'histogram2d', 'histogramdd', 'bincount', 'digitize', 'sum']

for method in array_reduce_methods:
    def create_method_template(method):
        def method_template(*args, **kwargs):
            deferred_array = args[0]
            return getattr(np, method)(deferred_array.eval(), *args[1:], **kwargs)
        return method_template
    setattr(api, method, create_method_template(method))


def where(cond, left, right):
    return Array(data=WhereOperation(cond.array_node,
                                     left.array_node,
                                     right.array_node,
                                     depth=python_max(left._depth, right._depth)))

