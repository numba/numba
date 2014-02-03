"""
LLVM pass that converts intrinsic into other math calls
"""
from __future__ import print_function, absolute_import
import llvm.core as lc


class DivmodFixer(object):
    """
    Fix 64-bit div/mod on 32-bit machines
    """
    NAMES = 'sdiv', 'udiv', 'srem', 'urem'
    I64 = lc.Type.int(64)

    def run(self, module):
        for func in module.functions:
            self.run_on_func(func)
        
    def run_on_func(self, func):
        to_replace = []
        for bb in func.basic_blocks:
            for instr in bb.instructions:
                opname = instr.opcode_name
                if opname in self.NAMES and instr.type == self.I64:
                    to_replace.append((instr, "numba.math.%s" % opname))

        if to_replace:
            builder = lc.Builder.new(func.entry_basic_block)
            for inst, name in to_replace:
                builder.position_before(inst)
                alt = self.declare(func.module, name)
                replacement = builder.call(alt, inst.operands)
                # fix replace_all_uses_with to not use ._ptr
                inst.replace_all_uses_with(replacement._ptr)
                inst.erase_from_parent()

    def declare(self, module, fname):
        fnty = lc.Type.function(self.I64, (self.I64, self.I64))
        fn = module.get_or_insert_function(fnty, name=fname)
        assert fn.is_declaration, ("%s is expected to be an intrinsic but "
                                   "it is defined" % fname)
        return fn


class IntrinsicMapping(object):
    def __init__(self, context, mapping=None, availintr=None):
        """
        Args
        ----
        mapping:
            Optional. Intrinsic name to alternative implementation.
            Default to global MAPPING

        availintr:
            Optional.  Available intrinsic set.
            Default to global AVAILINTR

        """
        self.context = context
        self.mapping = mapping or MAPPING
        self.availintr = availintr or AVAILINTR

    def run(self, module):
        self.apply_mapping(module)
        self.translate_intrinsic_to_cmath(module)

    def apply_mapping(self, module):
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

    def translate_intrinsic_to_cmath(self, module):
        for fn in self._iter_unavail(module):
            # Rename unavailable intrinsic to libc calls
            fn.name = INTR_TO_CMATH[fn.name]

        if __debug__:
            module.verify()

    def _iter_unavail(self, module):
        for fn in module.functions:
            if fn.is_declaration and fn.name.startswith('llvm.'):
                if fn.name not in self.availintr:
                    yield fn


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


AVAILINTR = ()


INTR_TO_CMATH = {
    "llvm.pow.f32": "powf",
    "llvm.pow.f64": "pow",

    "llvm.sin.f32": "sinf",
    "llvm.sin.f64": "sin",

    "llvm.cos.f32": "cosf",
    "llvm.cos.f64": "cos",

    "llvm.sqrt.f32": "sqrtf",
    "llvm.sqrt.f64": "sqrt",

    "llvm.exp.f32": "expf",
    "llvm.exp.f64": "exp",

    "llvm.log.f32": "logf",
    "llvm.log.f64": "log",

    "llvm.log10.f32": "log10f",
    "llvm.log10.f64": "log10",

    "llvm.fabs.f32": "fabsf",
    "llvm.fabs.f64": "fabs",

    "llvm.floor.f32": "floorf",
    "llvm.floor.f64": "floor",

    "llvm.ceil.f32": "ceilf",
    "llvm.ceil.f64": "ceil",
}

OTHER_CMATHS = '''
tan
tanf
sinh
sinhf
cosh
coshf
tanh
tanhf
asin
asinf
acos
acosf
atan
atanf
asinh
asinhf
acosh
acoshf
atanh
atanhf
expm1
expm1f
log1p
log1pf
log10
log10f
fmod
fmodf
'''.split()

INTR_MATH = frozenset(INTR_TO_CMATH.values()) | frozenset(OTHER_CMATHS)
