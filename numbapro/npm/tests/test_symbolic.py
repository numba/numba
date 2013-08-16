import dis
from numbapro.npm2.symbolic import SymbolicExecution
from .support import testcase, main


def foo(a, b, c, d, e, f, g):
    sum = a
    if a == b:
        for i in range(b):
            sum += i
    a = a + b - c / d // d % e
    a = a >> b << c | b ^ d & e
    b = +a
    b = -a
    b = ~a
    if a:
        return
    a = a > b and not a < c or a == d and b != e or a >= f or a <= g
    if a > b and not a < c or a == d and b != e or a >= f or a <= g:
        a = True
    else:
        a = False
        return sum

@testcase
def test_symbolic():
    dis.dis(foo)
    se = SymbolicExecution(foo)
    se.interpret()
    for blk in se.blocks:
        print blk


if __name__ == '__main__':
    main()
