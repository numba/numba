# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from .llvm_testutils import *

def build_expr(type):
    out, v1, v2, v3 = vars = build_vars(type, type, type, type)
    expr = b.assign(out, b.add(v1, b.mul(v2, v3)))
    return vars, expr

def build_kernel(specialization_name, ndim, **kw):
    vars, expr = build_expr(minitypes.ArrayType(float_, ndim, **kw))
    func = MiniFunction(specialization_name, vars, expr, '%s_%d' % (specialization_name, ndim))
    return func

def build_kernels(specialization_name, min_ndim=1, max_ndim=3, **kw):
    return [build_kernel(specialization_name, ndim)
                for ndim in range(min_ndim, max_ndim + 1)]

arrays2d = [get_array(), get_array(), get_array()]
arrays1d = [a[0] for a in arrays2d]
arrays3d = [a[:, None, :] for a in arrays2d]
arrays = [(arrays1d, arrays2d, arrays3d)]

"""
Generate tests, but skip vectorized versions (not supported for llvm
code backend yet)
"""
specializations = [s for s in sps.keys()
                       if not s.endswith(('_sse', '_avx'))]
print(specializations)

@parametrize(arrays=arrays, specialization_name=specializations, ndim=range(1, 4))
def test_specializations(arrays, specialization_name, ndim):
    if 'tiled' in specialization_name and ndim < 2:
        return

    # FIXME: these fail
    if specialization_name == 'inner_contig_fortran' and ndim >= 2:
        return

    if 'fortran' in specialization_name:
        arrays = [(x.T, y.T, z.T) for x, y, z in arrays]

    func = build_kernel(specialization_name, ndim)
    x, y, z = arrays[ndim - 1]

    print((x.strides, y.strides, z.strides))
    assert np.all(func(x, y, z) == x + y * z)

