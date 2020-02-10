import os
import unittest
import signal
import subprocess
import sys

from numba import errors
from numba.core.compiler_lock import (
    global_compiler_lock,
    require_global_compiler_lock,
)

from numba.tests.support import TestCase

linux_only = unittest.skipIf(not sys.platform.startswith('linux'), "linux only")


class TestCompilerLock(TestCase):
    def test_gcl_as_context_manager(self):
        with global_compiler_lock:
            require_global_compiler_lock()

    def test_gcl_as_decorator(self):
        @global_compiler_lock
        def func():
            require_global_compiler_lock()

        func()


@linux_only
class TestSignalHandler(TestCase):

    def test_sigabrt_caught(self):
        # emulate LLVM falling over with a SIGABRT
        with self.assertRaises(errors.FatalError) as raises:
            with global_compiler_lock:
                os.kill(os.getpid(), signal.SIGABRT)
        self.assertIn("The compilation chain has received SIGABRT",
                      str(raises.exception))

    def test_sigabrt_not_caught_if_permitted(self):
        code = """if 1:
            import os
            import signal
            from numba.core.compiler_lock import global_compiler_lock
            with global_compiler_lock:
                os.kill(os.getpid(), signal.SIGABRT)
        """
        env = dict(os.environ)
        env['NUMBA_DEBUG_PERMIT_SIGABRT'] = "1"
        popen = subprocess.Popen([sys.executable, "-c", code], env=env)
        popen.communicate()
        self.assertEqual(popen.returncode, -signal.SIGABRT)


if __name__ == '__main__':
    unittest.main()
