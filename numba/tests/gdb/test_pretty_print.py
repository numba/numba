# NOTE: This test is sensitive to line numbers as it checks breakpoints
from numba import njit
import numpy as np
from numba.tests.gdb_support import GdbMIDriver
from numba.tests.support import TestCase, needs_subprocess
import os
import unittest


@needs_subprocess
class Test(TestCase):

    def test(self):
        rdt = np.dtype([("x", np.int16, (2,)), ("y", np.float64)], align=True)

        @njit(debug=True)
        def foo():
            a = 1.234
            b = (1, 2, 3)
            c = ('a', b, 4)
            d = np.arange(5.)
            e = np.array([[1, 3j], [2, 4j]])
            f = "Some string" + "           L-Padded string".lstrip()
            g = 11 + 22j
            h = np.arange(24).reshape((4, 6))[::2, ::3]
            i = np.zeros(6, dtype=rdt)
            return a, b, c, d, e, f, g, h, i

        foo()

        extension = os.path.join('numba', 'misc', 'gdb_print_extension.py')
        driver = GdbMIDriver(__file__, init_cmds=['-x', extension], debug=True)
        driver.set_breakpoint(line=27)
        driver.run()
        driver.check_hit_breakpoint(1)

        # Ideally the function would be run to get the string repr of locals
        # but not everything appears in DWARF e.g. string literals. Further, str
        # on NumPy arrays seems to vary a bit in output. Therefore a regex based
        # match is used.
        #expect = ('TODO')
        driver.stack_list_variables(1)
        #driver.assert_regex_output(expect)
        driver.quit()


if __name__ == '__main__':
    unittest.main()
