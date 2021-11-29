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
        @njit(debug=True)
        def foo():
            a = 1.234
            b = (1, 2, 3)
            c = ('a', b, 4)
            d = np.arange(5.)
            e = np.array([[1, 3j], [2, 4j]])
            f = "Some string" + "           L-Padded string".lstrip()
            g = 11 + 22j
            return a, b, c, d, e, f, g

        foo()

        extension = os.path.join('numba', 'misc', 'gdb_print_extension.py')
        driver = GdbMIDriver(__file__, init_cmds=['-x', extension])
        driver.set_breakpoint(line=23)
        driver.run()
        driver.check_hit_breakpoint(1)

        # Ideally the function would be run to get the string repr of locals
        # but not everything appears in DWARF e.g. string literals. Further, str
        # on NumPy arrays seems to vary a bit in output. Therefore a regex based
        # match is used.
        expect = (r'[\{name="a",value="1.234"\},'
                  r'\{name="b",value="\(1, 2, 3\)"\},'
                  r'\{name="c",value="\(0x0, \(1, 2, 3\), 4)"\},'
                  r'\{name="d",value="\[\s+0. 1. 2. 3. 4.\]"\},'
                  # NOTE: output for variable e is split over these 2 lines
                  r'\{name="e,value="\[\[\s+1.+0.j\s+0.+3.j\]\\\n'
                  r'\s+\[\s+2.+0.j\s+0.+4.j\]\]"\},'
                  r'{name="f",value="\'Some stringL-Padded string\'"},'
                  r'{name="g",value="11+22j"}]')
        driver.stack_list_variables(1)
        driver.assert_regex_output(expect)
        driver.quit()


if __name__ == '__main__':
    unittest.main()
