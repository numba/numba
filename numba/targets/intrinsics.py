"""
LLVM pass that converts intrinsic into other math calls
"""
from __future__ import print_function, absolute_import
import llvmlite.llvmpy.core as lc
from llvmlite import ir


class _DivmodFixer(ir.Visitor):
    def visit_Instruction(self, instr):
        if instr.type == ir.IntType(64):
            if instr.opname in ['srem', 'urem', 'sdiv', 'udiv']:
                name = 'numba.math.{op}'.format(op=instr.opname)
                fn = self.module.globals.get(name)
                # Declare the function if it doesn't already exist
                if fn is None:
                    opty = instr.type
                    sdivfnty = ir.FunctionType(opty, [opty, opty])
                    fn = ir.Function(self.module, sdivfnty, name=name)
                # Replace the operation with a call to the builtin
                repl = ir.CallInstr(parent=instr.parent, func=fn,
                                    args=instr.operands, name=instr.name)
                instr.parent.replace(instr, repl)


def fix_divmod(mod):
    """Replace division and reminder instructions to builtins calls
    """
    _DivmodFixer().visit(mod)


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
            # Ignore unrecognized llvm intrinsic
            fn.name = INTR_TO_CMATH.get(fn.name, fn.name)

        if __debug__:
            module.verify()

    def _iter_unavail(self, module):
        for fn in module.functions:
            if fn.is_declaration and fn.name.startswith('llvm.'):
                if fn.name not in self.availintr:
                    yield fn


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

    "llvm.trunc.f32": "truncf",
    "llvm.trunc.f64": "trunc",
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
atan2
atan2f
atan2_fixed
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
round
roundf
'''.split()

INTR_MATH = frozenset(INTR_TO_CMATH.values()) | frozenset(OTHER_CMATHS)
