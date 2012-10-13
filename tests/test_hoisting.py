"""
Test loop-invariant code motion. Write more tests with different associations.

NOTE: most of the tests are part of Cython:
    https://github.com/markflorisson88/cython/tree/_array_expressions/tests/array_expressions
"""

from testutils import *

import pytest

cinner = sps['inner_contig']

@pytest.mark.skipif('not xmldumper.have_lxml')
def test_hoist():
    """
    >> test_hoist()
    """
    type1 = double[:, :]
    type2 = double[:, :]
    type1.broadcasting = (False, False)
    type2.broadcasting = (False, True)

    var1, var2 = vars = build_vars(type1, type2)
    expr = b.add(var1, b.add(var2, var2))
    expr = b.add(var1, var2)
    body = b.assign(var1, expr)
    func = build_function(vars, body)

    result_ast, code_output = specialize(cinner, func)
    e = toxml(result_ast)
    assert e.xpath('not(//NDIterate)')

    # Check the loop level of the hoisted expression
    op1, op2 = e.xpath(
        '//FunctionNode//ArrayFunctionArgument/DataPointer/@value')
    broadcasting_pointer_temp, = e.xpath(
        '//AssignmentExpr[./rhs/DataPointer[@value="%s"]]/lhs/TempNode/@value' % op2)

    q = '//ForNode[.//AssignmentExpr/rhs//TempNode[@value="%s"]]/@loop_level'
    loop_level, = e.xpath(q % broadcasting_pointer_temp)
    assert loop_level == "0", loop_level

def test_hoist_3d():
    """
    >>> test_hoist_3d()
    """
    type1 = npy_intp[:, :, :]
    type2 = npy_intp[:, :, :]
    type3 = npy_intp[:, :, :]
    type1.broadcasting = (False, True, True)
    type2.broadcasting = (True, False, True)
    type3.broadcasting = (True, True, False)

    out_type = npy_intp[:, :, :]

    out, var1, var2, var3 = vars = build_vars(out_type, type1, type2, type3)
    #v1 = b.mul(var1, var2)
    #v2 = b.mul(var2, var2)
    #v3 = b.mul(var3, var3)
    expr = b.mul(b.mul(var1, var2), var3)
    body = b.assign(out, expr)
    func = build_function(vars, body)

    result_ast, code_output = specialize(cinner, func)


if __name__ == '__main__':
    import doctest
    doctest.testmod()