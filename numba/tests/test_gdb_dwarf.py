"""Tests for gdb interacting with the DWARF numba generates"""
from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect
import unittest


@linux_only
@needs_gdb
@skip_unless_pexpect
class TestGDBDwarf(TestCase):
    # This runs the tests in numba.tests.gdb, each submodule must contain one
    # test class called "Test" and it must contain one test called "test".
    # Variation is provided by the module name. The reason this convention exits
    # is because gdb tests tend to be line number sensitive (breakpoints etc
    # care about this) and doing this prevents constant churn and permits the
    # reuse of the existing subprocess_test_runner harness.
    _NUMBA_OPT_0_ENV = {'NUMBA_OPT': '0'}

    def _subprocess_test_runner(self, test_mod):
        themod = f'numba.tests.gdb.{test_mod}'
        self.subprocess_test_runner(test_module=themod,
                                    test_class='Test',
                                    test_name='test',
                                    envvars=self._NUMBA_OPT_0_ENV)

    def test_basic(self):
        self._subprocess_test_runner('test_basic')

    def test_array_arg(self):
        self._subprocess_test_runner('test_array_arg')

    def test_conditional_breakpoint(self):
        self._subprocess_test_runner('test_conditional_breakpoint')

    def test_break_on_symbol(self):
        self._subprocess_test_runner('test_break_on_symbol')


if __name__ == '__main__':
    unittest.main()
