from __future__ import print_function
from pprint import pprint
from numba import bytecode, interpreter, typing, typeinfer, lowering, targets


class Flags(object):
    FLAGS_SET = frozenset(['enable_pyobject'])

    def __init__(self):
        self._enabled = set()

    def set(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.add(name)

    def unset(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.discard(name)

    def __getattr__(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        return name in self._enabled

    def __repr__(self):
        return "Flags(%s)" % ', '.join(str(x) for x in self._enabled)

DEFAULT_FLAGS = Flags()


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = targets.CPUContext()
    cfunc = compile_extra(typingctx, targetctx, func, args, return_type, flags)
    return targetctx, cfunc


def compile_extra(typingctx, targetctx, func, args, return_type, flags):
    # Translate to IR
    interp = translate_stage(func)

    # Type inference
    try:
        typemap, restype, calltypes = type_inference_stage(typingctx, interp,
                                                           args, return_type)
    except Exception, e:
        print(flags)
        if not flags.enable_pyobject:
            raise
        # TODO handle the fail reason
        fail_reason = e
        return py_lowering_stage(targetctx, interp)
    else:
        return native_lowering_stage(targetctx, interp, typemap, restype,
                                     calltypes)


def translate_stage(func):
    bc = bytecode.ByteCode(func=func)
    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump()

    if __debug__:
        for syn in interp.syntax_info:
            print(syn)

    interp.verify()
    return interp


def type_inference_stage(typingctx, interp, args, return_type):
    infer = typeinfer.TypeInferer(typingctx, interp.blocks)

    # Seed argument types
    for arg, ty in zip(interp.argspec.args, args):
        infer.seed_type(arg, ty)

    # Seed return type
    if return_type is not None:
        infer.seed_return(return_type)

    infer.build_constrain()
    infer.propagate()
    typemap, restype, calltypes = infer.unify()

    if __debug__:
        pprint(typemap)
        pprint(restype)
        pprint(calltypes)

    return typemap, restype, calltypes


def native_lowering_stage(targetctx, interp, typemap, restype, calltypes):
    # Lowering
    fndesc = lowering.describe_function(interp, typemap, restype, calltypes)

    lower = lowering.Lower(targetctx, fndesc)
    lower.lower()

    if __debug__:
        print(lower.module)

    # Prepare for execution
    cfunc = targetctx.get_executable(lower.function, fndesc)
    return cfunc


def py_lowering_stage(targetctx, interp):
    lower = lowering.PyLower(targetctx, interp)
    lower.lower()

    if __debug__:
        print(lower.module)

    raise NotImplementedError
