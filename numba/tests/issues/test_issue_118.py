# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import autojit

@autojit
def get_jenks_breaks(data, lower_class_limits, n_classes):
    k = int(len(data) - 1)
    for countNum in range(n_classes):
        # problem inferring `k` in slice
        k = int(lower_class_limits[k, countNum] - 1)

def test():
    n_classes = 5
    data = np.ones(10)
    lower_class_limits = np.empty((data.size + 1, n_classes + 1))
    #
    #        File "/Users/sklam/dev/numba/numba/type_inference/infer.py", line 1055, in visit_Subscript
    #    if slice_type.variable.type.is_unresolved:
    #        File "/Users/sklam/dev/numba/numba/minivect/minitypes.py", line 492, in __getattr__
    #            return getattr(type(self), attr)
    #AttributeError: type object 'tuple_' has no attribute 'variable'
    print((get_jenks_breaks(data, lower_class_limits, n_classes)))

if __name__ == '__main__':
    test()
