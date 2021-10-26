"""Helpers for running gdb related testing"""
import os
import re
import sys
import unittest
from numba.core import config
from numba.misc.gdb_hook import _confirm_gdb

# check if gdb is present and working
try:
    _confirm_gdb(need_ptrace_attach=False) # The driver launches as `gdb EXE`.
    _HAVE_GDB = True
except Exception:
    _HAVE_GDB = False

_msg = "functioning gdb with correct ptrace permissions is required"
needs_gdb = unittest.skipUnless(_HAVE_GDB, _msg)

try:
    import pexpect
    _HAVE_PEXPECT = True
except ImportError:
    _HAVE_PEXPECT = False


_msg = "pexpect module needed for test"
skip_unless_pexpect = unittest.skipUnless(_HAVE_PEXPECT, _msg)


class GdbMIDriver(object):
    """
    Driver class for the GDB machine interface:
    https://sourceware.org/gdb/onlinedocs/gdb/GDB_002fMI.html
    """
    def __init__(self, file_name, debug=False, timeout=120):
        if not _HAVE_PEXPECT:
            msg = ("This driver requires the pexpect module. This can be "
                   "obtained via:\n\n$ conda install pexpect")
            raise RuntimeError(msg)
        if not _HAVE_GDB:
            msg = ("This driver requires a gdb binary. This can be "
                   "obtained via the system package manager.")
            raise RuntimeError(msg)
        self._gdb_binary = config.GDB_BINARY
        self._python = sys.executable
        self._debug = debug
        self._file_name = file_name
        self._timeout = timeout
        self._drive()

    def _drive(self):
        """This function sets up the caputured gdb instance"""
        assert os.path.isfile(self._file_name)
        cmd = [self._gdb_binary, '--interpreter', 'mi', '--args', self._python,
               self._file_name]
        self._captured = pexpect.spawn(' '.join(cmd))
        if self._debug:
            self._captured.logfile = sys.stdout.buffer

    def _captured_expect(self, expect):
        try:
            self._captured.expect(expect, timeout=self._timeout)
        except pexpect.exceptions.TIMEOUT as e:
            msg = f"Expected value did not arrive: {expect}."
            raise ValueError(msg) from e

    def assert_output(self, expected):
        """Asserts that the current output string contains the expected."""
        output = self._captured.after
        decoded = output.decode('utf-8')
        assert expected in decoded, f'decoded={decoded}\nexpected={expected})'

    def assert_regex_output(self, expected):
        """Asserts that the current output string contains the expected
        regex."""
        output = self._captured.after
        decoded = output.decode('utf-8')
        found = re.search(expected, decoded)
        assert found, f'decoded={decoded}\nexpected={expected})'

    def _run_command(self, command, expect=''):
        self._captured.sendline(command)
        self._captured_expect(expect)

    def run(self):
        """gdb command ~= 'run'"""
        self._run_command('-exec-run', expect=r'\^running.*\r\n')

    def cont(self):
        """gdb command ~= 'continue'"""
        self._run_command('-exec-continue', expect=r'\^running.*\r\n')

    def quit(self):
        """gdb command ~= 'quit'"""
        self._run_command('-gdb-exit', expect=r'-gdb-exit')
        self._captured.terminate()

    def next(self):
        """gdb command ~= 'next'"""
        self._run_command('-exec-next', expect=r'\*stopped,.*\r\n')

    def step(self):
        """gdb command ~= 'step'"""
        self._run_command('-exec-step', expect=r'\*stopped,.*\r\n')

    def set_breakpoint(self, line=None, symbol=None, condition=None):
        """gdb command ~= 'break'"""
        if line is not None and symbol is not None:
            raise ValueError("Can only supply one of line or symbol")
        bp = '-break-insert '
        if condition is not None:
            bp += f'-c "{condition}" '
        if line is not None:
            assert isinstance(line, int)
            bp += f'-f {self._file_name}:{line} '
        if symbol is not None:
            assert isinstance(symbol, str)
            bp += f'-f {symbol} '
        self._run_command(bp, expect=r'\^done')

    def check_hit_breakpoint(self, number=None, line=None):
        """Checks that a breakpoint has been hit"""
        self._captured_expect(r'\*stopped,.*\r\n')
        self.assert_output('*stopped,reason="breakpoint-hit",')
        if number is not None:
            assert isinstance(number, int)
            self.assert_output(f'bkptno="{number}"')
        if line is not None:
            assert isinstance(line, int)
            self.assert_output(f'line="{line}"')

    def stack_list_arguments(self, print_values=1, low_frame=0, high_frame=0):
        """gdb command ~= 'info args'"""
        for x in (print_values, low_frame, high_frame):
            assert isinstance(x, int) and x in (0, 1, 2)
        cmd = f'-stack-list-arguments {print_values} {low_frame} {high_frame}'
        self._run_command(cmd, expect=r'\^done,.*\r\n')

    def stack_list_variables(self, print_values=1):
        """gdb command ~= 'info locals'"""
        assert isinstance(print_values, int) and print_values in (0, 1, 2)
        cmd = f'-stack-list-variables {print_values}'
        self._run_command(cmd, expect=r'\^done,.*\r\n')
