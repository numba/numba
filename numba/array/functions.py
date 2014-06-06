from array import Array
import numpy as np
from nodes import WhereOperation

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
array_statistics_methods = ['amin', 'amax', 'nanmin', 'nanmax', 'ptp',
    'percentile', 'median', 'average', 'mean', 'std', 'var', 'nanmean',
    'nanstd', 'nanvar', 'corrcoef', 'correlate', 'cov', 'histogram',
    'histogram2d', 'histogramdd', 'bincount', 'digitize']

for method in array_statistics_methods:
    def create_method_template(method):
        def method_template(*args, **kwargs):
            deferred_array = args[0]
            return getattr(np, method)(deferred_array.eval(), *args[1:], **kwargs)
        return method_template
    setattr(api, method, create_method_template(method))


def where(cond, left, right):
    return Array(data=WhereOperation(cond.array_node,
                                     left.array_node,
                                     right.array_node))

