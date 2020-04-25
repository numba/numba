import platform
from unittest.mock import NonCallableMock
from itertools import chain
from datetime import datetime
from contextlib import redirect_stdout
from io import StringIO

from numba.tests.support import TestCase
import numba.misc.numba_sysinfo as nsi


class TestSysInfo(TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestSysInfo, cls).setUpClass()
        cls.info = nsi.get_sysinfo()
        cls.safe_contents = {
            int: (
                nsi._cpu_count,
            ),
            float: (
                nsi._runtime,
            ),
            str: (
                nsi._machine,
                nsi._cpu_name,
                nsi._platform_name,
                nsi._os_name,
                nsi._os_version,
                nsi._python_comp,
                nsi._python_impl,
                nsi._python_version,
                nsi._llvm_version,
            ),
            bool: (
                nsi._cu_dev_init,
                nsi._roc_available,
                nsi._svml_state,
                nsi._svml_loaded,
                nsi._svml_operational,
                nsi._llvm_svml_patched,
                nsi._tbb_thread,
                nsi._openmp_thread,
                nsi._wkq_thread,
            ),
            list: (
                nsi._roc_toolchains,
                nsi._errors,
                nsi._warnings,
            ),
            dict: (
                nsi._numba_env_vars,
            ),
            datetime: (
                nsi._start,
            ),
        }
        cls.safe_keys = chain(*cls.safe_contents.values())

    @classmethod
    def tearDownClass(cls):
        super(TestSysInfo, cls).tearDownClass()
        # System info might contain long strings or lists so delete it.
        del cls.info

    def test_has_safe_keys(self):
        for k in self.safe_keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info)

    def test_safe_content_type(self):
        for t, keys in self.safe_contents.items():
            for k in keys:
                with self.subTest(k=k):
                    self.assertIsInstance(self.info[k], t)

    def test_has_no_error(self):
        self.assertFalse(self.info[nsi._errors])

    def test_display_empty_info(self):
        output = StringIO()
        with redirect_stdout(output):
            res = nsi.display_sysinfo({})
        self.assertIsNone(res)
        output.close()


class TestSysInfoWithPsutil(TestCase):

    mem_total = 2 * 1024 ** 2  # 2_097_152
    mem_available = 1024 ** 2  # 1_048_576
    cpus_list = [1, 2]

    @classmethod
    def setUpClass(cls):
        super(TestSysInfoWithPsutil, cls).setUpClass()
        cls.psutil_orig_state = nsi._psutil_import
        # Mocking psutil
        nsi._psutil_import = True
        nsi.psutil = NonCallableMock()
        vm = nsi.psutil.virtual_memory.return_value
        vm.total = cls.mem_total
        vm.available = cls.mem_available
        proc = nsi.psutil.Process.return_value
        proc.cpu_affinity.return_value = cls.cpus_list

        cls.info = nsi.get_os_spec_info(platform.system())

    @classmethod
    def tearDownClass(cls):
        super(TestSysInfoWithPsutil, cls).tearDownClass()
        nsi._psutil_import = cls.psutil_orig_state

    def test_has_all_data(self):
        keys = (nsi._mem_total, nsi._mem_available, nsi._cpus_allowed)
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)
        self.assertIn(nsi._cpus_list, self.info.keys())
        self.assertIsInstance(self.info[nsi._cpus_list], str)

    def test_has_correct_values(self):
        self.assertEqual(self.info[nsi._mem_total], self.mem_total)
        self.assertEqual(self.info[nsi._mem_available], self.mem_available)
        self.assertEqual(self.info[nsi._cpus_allowed], len(self.cpus_list))
        self.assertEqual(self.info[nsi._cpus_list],
                         ' '.join(str(n) for n in self.cpus_list))


class TestSysInfoWithoutPsutil(TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestSysInfoWithoutPsutil, cls).setUpClass()
        cls.psutil_orig_state = nsi._psutil_import
        nsi._psutil_import = False
        cls.info = nsi.get_os_spec_info(platform.system())

    @classmethod
    def tearDownClass(cls):
        super(TestSysInfoWithoutPsutil, cls).tearDownClass()
        nsi._psutil_import = cls.psutil_orig_state

    def test_has_all_data(self):
        keys = (nsi._mem_total, nsi._mem_available)
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())
                self.assertIsInstance(self.info[k], int)


class TestPlatformSpecificInfo(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.plat_spec_info = {
            'Linux': {
                str: (nsi._libc_version,),
            },
            'Windows': {
                str: (nsi._os_spec_version,),
            },
            'Darwin': {
                str: (nsi._os_spec_version,),
            },
        }
        cls.os_name = platform.system()
        cls.contents = cls.plat_spec_info.get(cls.os_name, {})
        cls.info = nsi.get_os_spec_info(cls.os_name)

    def test_has_all_data(self):
        keys = chain(*self.contents.values())
        for k in keys:
            with self.subTest(k=k):
                self.assertIn(k, self.info.keys())

    def test_content_type(self):
        for t, keys in self.contents.items():
            for k in keys:
                with self.subTest(k=k):
                    self.assertIsInstance(self.info[k], t)
