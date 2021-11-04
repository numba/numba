from .test_cli import run_cmd
import json
from numba.tests.support import TestCase, override_env_config
import os
import sys
from tempfile import TemporaryDirectory
import unittest

class TestChromeTraceModule(TestCase):
    """
    Test chrome tracing generated file(s).
    """

    def test_trace_output(self):
        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_trace.json")
            with override_env_config('NUMBA_CHROME_TRACE', path):
                from .chrome_trace_usecase import __file__
                cmdline = [sys.executable, __file__]
                run_cmd(cmdline)
                with open(path) as file:
                    jfile = json.load(file)
                    self.assertTrue(jfile)