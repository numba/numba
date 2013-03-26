from numba import *
import numba as nb

@autojit
def test_count(type):

    ttuple = nb.typedtuple(type, [1, 2, 3, 4, 5, 1, 2])
    return ttuple.count(0), ttuple.count(3), ttuple.count(1)

def test(module):
    assert test_count(int_) == (0, 1, 2)

if __name__ == "__main__":
    import __main__ as module
else:
    import test_typed_tuple as module

test(module)
__test__ = {}
