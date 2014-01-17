"""
LLVM pass that converts intrinsic into other math calls
"""
from __future__ import print_function, absolute_import
import llvm.core as lc


class IntrinsicMapping(object):
    def __init__(self, context, mapping=None):
        self.context = context
        self.mapping = mapping or MAPPING

    def run(self, module):
        modified = []
        for fn in module.functions:
            if fn.is_declaration and fn.name in self.mapping:
                imp = self.mapping[fn.name]
                imp(self.context, fn)
                modified.append(fn)

        # Rename all modified functions
        for fn in modified:
            fn.name = "numba." + fn.name

        if __debug__:
            module.verify()


def powi_as_pow(context, fn):
    builder = lc.Builder.new(fn.append_basic_block(""))
    x, y = fn.args
    fy = builder.sitofp(y, x.type)
    pow = lc.Function.intrinsic(fn.module, lc.INTR_POW, [x.type])
    builder.ret(builder.call(pow, (x, fy)))


MAPPING = {
    "llvm.powi.f32": powi_as_pow,
    "llvm.powi.f64": powi_as_pow,
}

