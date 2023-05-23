import itertools
from llvmlite.binding import view_dot_graph, get_function_cfg

from numba.core.frontend2 import bcinterp, bc2rvsdg
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS, Flags
from numba.core import typing, cpu, types
from numba.tests import usecases


DEFAULT_FLAGS = Flags()
DEFAULT_FLAGS.nrt = True


def compile(func):
    func_ir = bcinterp.run_frontend(func)
    args = (types.intp, types.intp)
    return_type = None

    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    flags = DEFAULT_FLAGS
    cres = compile_ir(
        typingctx, targetctx, func_ir, args, return_type, flags,
        locals={},
    )
    llmod = cres.library._final_module
    fn = llmod.get_function(cres.fndesc.mangled_name)
    if bc2rvsdg.DEBUG_GRAPH:
        view_dot_graph(get_function_cfg(fn)).view()
    return cres.entry_point


# Working tests ---------------------------------------------------------------

def test_sum1d():
    pyfunc = usecases.sum1d
    cfunc = compile(pyfunc)
    ss = -1, 0, 1, 100, 200
    es = -1, 0, 1, 100, 200

    for args in itertools.product(ss, es):
        assert pyfunc(*args) == cfunc(*args)


def test_sum2d():
    pyfunc = usecases.sum2d
    cfunc = compile(pyfunc)
    ss = -1, 0, 1, 100, 200
    es = -1, 0, 1, 100, 200

    for args in itertools.product(ss, es):
        assert pyfunc(*args) == cfunc(*args)


def test_x_or_y():
    pyfunc = lambda x, y: x or y
    cfunc = compile(pyfunc)
    xx = True, False
    yy = True, False
    for args in itertools.product(xx, yy):
        assert pyfunc(*args) == cfunc(*args)
    xx = 0, 1, 2
    yy = 0, 1, 2
    for args in itertools.product(xx, yy):
        assert pyfunc(*args) == cfunc(*args)


def test_andor():
    pyfunc = usecases.andor
    cfunc = compile(pyfunc)
    # Argument boundaries
    xs = -1, 0, 1, 9, 10, 11
    ys = -1, 0, 1, 9, 10, 11

    for args in itertools.product(xs, ys):
        assert pyfunc(*args) == cfunc(*args)


# Not yet working... ----------------------------------------------------------



# if __name__ == "__main__":
#     test_andor()