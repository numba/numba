import unittest

from numba import cuda, jit, njit


@jit
def add1(a: int, b: int) -> int:
    """Add two integers
    Args:
        a (int): one integer
        b (int): the other integer
    Returns:
        int: the sum
    """
    return a + b


@njit
def add2(a: int, b: int) -> int:
    """Add two integers
    Args:
        a (int): one integer
        b (int): the other integer
    Returns:
        int: the sum
    """
    return a + b


@cuda.jit
def add3(a: int, b: int) -> int:
    """Add two integers
    Args:
        a (int): one integer
        b (int): the other integer
    Returns:
        int: the sum
    """
    return a + b


DOC = """Add two integers
    Args:
        a (int): one integer
        b (int): the other integer
    Returns:
        int: the sum
    """


class TestCooperativeGroups(unittest.TestCase):
    def test_doc_decorated_functions(self):
        """
        checks that the decorated function preserve the original __doc__
        """
        self.assertEqual(DOC, add1.__doc__)
        self.assertEqual(DOC, add2.__doc__)
        self.assertEqual(DOC, add3.__doc__)

    def test_module_decorated_functions(self):
        """
        checks that the decorated function preserve the original __module__
        """
        self.assertEqual(__name__, add1.__module__)
        self.assertEqual(__name__, add2.__module__)
        self.assertEqual(__name__, add3.__module__)


if __name__ == '__main__':
    unittest.main()
