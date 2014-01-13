# -*- coding: utf-8 -*-

'''
This bug is nondeterministic.  Run it a few times.
'''

from __future__ import print_function, division, absolute_import

import numpy as np

from numba import autojit, typeof, int32

@autojit
def jenks_matrices_no_return(data):
    variance = 0.0
    for l in range(2, len(data) + 1):
        sum = 0.0
        sum_squares = 0.0
        w = 0.0
        for m in range(1, l + 1):
            lower_class_limit = l - m + 1
            val = data[lower_class_limit - 1]
            w += 1
            sum += val
            sum_squares += val * val
            variance = sum_squares - (sum * sum) / w

@autojit
def jenks_matrices_with_return(data):
    variance = 0.0
    for l in range(2, len(data) + 1):
        sum = 0.0
        sum_squares = 0.0
        w = 0.0
        for m in range(1, l + 1):
            lower_class_limit = l - m + 1
            val = data[lower_class_limit - 1]
            w += 1
            sum += val
            sum_squares += val * val
            variance = sum_squares - (sum * sum) / w
    return variance

def test():
    data = np.empty(10, dtype=np.float64)

    #        File "/Users/sklam/dev/numba/numba/type_inference/infer.py", line 493, in resolve_variable_types
    #    start_point.simplify()
    #        File "/Users/sklam/dev/numba/numba/typesystem/ssatypes.py", line 625, in simplify
    #            assert False
    jenks_matrices_no_return(data)


    #       File "/Users/sklam/dev/numba/numba/type_inference/infer.py", line 495, in resolve_variable_types
    #    self.remove_resolved_type(start_point)
    #        File "/Users/sklam/dev/numba/numba/type_inference/infer.py", line 393, in remove_resolved_type
    #            assert not type.is_scc
    jenks_matrices_with_return(data)

if __name__ == "__main__":
    test()

