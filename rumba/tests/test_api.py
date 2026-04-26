import pytest

import rumba
from rumba import RumbaUnsupportedError


def test_import_and_version():
    assert rumba.__version__


def test_njit_direct_decorator_executes_scalar_add():
    @rumba.njit
    def add(a, b):
        return a + b

    assert add.py_func(1, 2) == 3
    assert add(1, 2) == 3
    assert add.signatures


def test_njit_call_decorator_executes_float_branch():
    @rumba.njit(cache=True, debug=True)
    def choose(a, b):
        if a > b:
            return a - b
        return b - a

    assert choose(1.5, 4.0) == pytest.approx(2.5)
    assert "if" in choose.inspect_c()


def test_jit_alias():
    @rumba.jit
    def add(a, b):
        return a + b

    assert add(4, 5) == 9


def test_explicit_signature():
    @rumba.njit(signature=("int64", "int64"))
    def add(a, b):
        return a + b

    assert add(2, 7) == 9


def test_unsupported_option_raises():
    with pytest.raises(RumbaUnsupportedError, match="unsupported njit option"):
        rumba.njit(parallel=True)


def test_loop_execution():
    @rumba.njit
    def total(n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    assert total(6) == 15


def test_inspection_helpers():
    @rumba.njit
    def add(a, b):
        return a + b

    bytecode = add.inspect_bytecode()
    ast_summary = add.inspect_rumba_ast()
    assert any(inst["opname"] == "RETURN_VALUE" for inst in bytecode)
    assert ast_summary["name"] == "add"
    assert ast_summary["body"] == ["Return"]


def test_unsupported_list_argument_raises():
    @rumba.njit
    def first(x):
        return x[0]

    with pytest.raises(RumbaUnsupportedError, match="unsupported argument type"):
        first([1])
