import platform
from unittest.mock import Mock
from itertools import chain
from datetime import datetime

from numba.tests.support import TestCase
import numba.misc.numba_sysinfo as sysinfo


class TestSysInfo(TestCase):

    def setUp(self):
        self.info = sysinfo.get_sysinfo()
        self.min_contents = {
            int: (
                'CPU Count',
            ),
            float: (
                'Runtime',
            ),
            str: (
                'Machine',
                'CPU Name',
                'Platform Name',
                'OS Name',
                'OS Version',
                'Python Compiler',
                'Python Implementation',
                'Python Version',
                'LLVM Version',
            ),
            bool: (
                'CUDA Device Init',
                'ROC Available',
                'SVML State',
                'SVML Lib Loaded',
                'SVML Operational',
                'LLVM SVML Patched',
                'TBB Threading',
                'OpenMP Threading',
                'Workqueue Threading',
            ),
            list: (
                'ROC Toolchains',
                'Errors',
                'Warnings',
            ),
            dict: (
                'Numba Env Vars',
            ),
            datetime: (
                'Start',
            ),
        }

    def test_has_minimal_keys(self):
        min_key_set = chain(*self.min_contents.values())
        for k in min_key_set:
            with self.subTest(k=k):
                self.assertIn(k, self.info)

    def test_content_type(self):
        for t, keys in self.min_contents.items():
            for k in keys:
                with self.subTest(k=k):
                    self.assertIsInstance(self.info[k], t)


class TestSysInfoWithPsutil(TestCase):

    def setUp(self):
        # Mocking psutil
        sysinfo._psutil_import = True
        sysinfo.psutil = Mock()
        vm = sysinfo.psutil.virtual_memory.return_value
        vm.total = 2 * 1024 ** 3
        vm.available = 1024 ** 3
        proc = sysinfo.psutil.Process.return_value
        proc.cpu_affinity.return_value = [1, 2]

        self.info = sysinfo.get_os_spec_info(platform.system())

    def test_has_all_data(self):
        keys = ('Mem Total', 'Mem Available', 'CPUs Allowed')
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)
        self.assertIn('List CPUs Allowed', self.info.keys())
        self.assertIsInstance(self.info['List CPUs Allowed'], str)


class TestSysInfoWithoutPsutil(TestCase):

    def setUp(self):
        sysinfo._psutil_import = False
        self.info = sysinfo.get_os_spec_info(platform.system())

    def test_has_all_data(self):
        keys = ('Mem Total', 'Mem Available')
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)
