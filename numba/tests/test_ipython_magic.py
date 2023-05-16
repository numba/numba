from unittest import TestCase
from IPython.terminal.interactiveshell import TerminalInteractiveShell

code = u"""\
import numba

@numba.njit
def f(x):
    return 2*x

x = f(41.0)
"""


class IPythonMagic(TestCase):

    def test_numba(self):
        ip = TerminalInteractiveShell()
        ip.extension_manager.load_extension('numba')
        ip.run_cell_magic("numba", "", code)
        ip.ex(f'{code}')
        self.assertEqual(ip.user_ns['x'], 82.0)
