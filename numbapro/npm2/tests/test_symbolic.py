from numbapro.npm2.symbolic import SymbolicExecution
from .support import testcase, main


def foo(a, b):
    c = a + b
    if a > b:
        return c + a - b
    else:
        sum = c
        for i in range(b - a):
            sum += i
        return sum

@testcase
def test_symbolic():
    se = SymbolicExecution(foo)
    se.visit()

if __name__ == '__main__':
    main()
