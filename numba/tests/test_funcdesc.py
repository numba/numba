
import unittest
from numba import njit


class TestModule(unittest.TestCase):
    def test_module_not_in_namespace(self):
        """ Test of trying to run a compiled function
        where the module from which the function is being compiled
        doesn't exist in the namespace.
        """
        filename = 'test.py'
        name = 'mypackage'
        code = """
def f(x):
    return x
"""

        objs = dict(__file__=filename, __name__=name)
        compiled = compile(code, filename, 'exec')
        exec(compiled, objs)

        compiled_f = njit(objs['f'])
        with self.assertRaises(ModuleNotFoundError) as raises:
            compiled_f(0)
        msg = "can't compile f: import of module mypackage failed"
        self.assertIn(msg, str(raises.exception))


if __name__ == '__main__':
    unittest.main()
