from ..compiler import compile
from ..types import intp
from .support import testcase, main, assertTrue

def enumerate_range():
    tmp = 0
    for i, j in enumerate(range(10, 20)):
        tmp += i * j
    return tmp

@testcase
def test_enumerate_range():
    cfunc = compile(enumerate_range, intp, ())
    print cfunc(), enumerate_range()
    assertTrue(cfunc() == enumerate_range())

if __name__ == '__main__':
    main()
