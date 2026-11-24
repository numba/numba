import os
import tempfile
from textwrap import dedent
import unittest
from unittest import mock
from numba.tests.support import (TestCase, temp_directory, override_env_config,
                                 run_in_subprocess)
from numba.core import config, vector_library

try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

_skip_msg = "pyyaml needed for configuration file tests"
needs_yaml = unittest.skipIf(not _HAVE_YAML, _skip_msg)


@needs_yaml
class TestConfig(TestCase):

    # Disable parallel testing due to envvars modification
    _numba_parallel_test_ = False

    def setUp(self):
        # use support.temp_directory, it can do the clean up
        self.tmppath = temp_directory('config_tmp')
        self.maxDiff = 2500
        super(TestConfig, self).setUp()

    def mock_cfg_location(self):
        """
        Creates a mock launch location.
        Returns the location path.
        """
        return tempfile.mkdtemp(dir=self.tmppath)

    def inject_mock_cfg(self, location, cfg):
        """
        Injects a mock configuration at 'location'
        """
        tmpcfg = os.path.join(location, config._config_fname)
        with open(tmpcfg, 'wt') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    def get_settings(self):
        """
        Gets the current numba config settings
        """
        store = dict()
        for x in dir(config):
            if x.isupper():
                store[x] = getattr(config, x)
        return store

    def create_config_effect(self, cfg):
        """
        Returns a config "original" from a location with no config file
        and then the impact of applying the supplied cfg dictionary as
        a config file at a location in the returned "current".
        """

        # store original cwd
        original_cwd = os.getcwd()

        # create mock launch location
        launch_dir = self.mock_cfg_location()

        # switch cwd to the mock launch location, get and store settings
        os.chdir(launch_dir)
        # use override to ensure that the config is zero'd out with respect
        # to any existing settings
        with override_env_config('_', '_'):
            original = self.get_settings()

        # inject new config into a file in the mock launch location
        self.inject_mock_cfg(launch_dir, cfg)

        try:
            # override something but don't change the value, this is to refresh
            # the config and make sure the injected config file is read
            with override_env_config('_', '_'):
                current = self.get_settings()
        finally:
            # switch back to original dir with no new config
            os.chdir(original_cwd)
        return original, current

    def test_config(self):
        # ensure a non empty settings file does impact config and that the
        # case of the key makes no difference
        key = 'COLOR_SCHEME'
        for case in [str.upper, str.lower]:
            orig, curr = self.create_config_effect({case(key): 'light_bg'})
            self.assertTrue(orig != curr)
            self.assertTrue(orig[key] != curr[key])
            self.assertEqual(curr[key], 'light_bg')
            # check that just the color scheme is the cause of difference
            orig.pop(key)
            curr.pop(key)
            self.assertEqual(orig, curr)

    def test_empty_config(self):
        # ensure an empty settings file does not impact config
        orig, curr = self.create_config_effect({})
        self.assertEqual(orig, curr)

    @unittest.skipUnless(config.ENABLE_AVX,
                         "test expects NUMBA_ENABLE_AVX==True")
    def test_nocona_disables_avx(self):
        # test with nocona
        new_env = os.environ.copy()
        new_env.pop('NUMBA_ENABLE_AVX', None)  # clear NUMBA_ENABLE_AVX

        new_env['NUMBA_CPU_NAME'] = 'nocona'
        code = ("from numba.core import config\n"
                "print('---->', bool(config.ENABLE_AVX))\n"
                "assert not config.ENABLE_AVX")
        out, err = run_in_subprocess(dedent(code), env=new_env)
        err_msg = err.decode('utf-8')
        out_msg = out.decode('utf-8')
        ex_expected = "----> False"
        self.assertIn(ex_expected, out_msg, msg=err_msg)

        # test with skylake-avx512
        new_env['NUMBA_CPU_NAME'] = 'skylake-avx512'
        code = ("from numba.core import config\n"
                "print('---->', bool(config.ENABLE_AVX))\n"
                "assert config.ENABLE_AVX")
        out, err = run_in_subprocess(dedent(code), env=new_env)
        err_msg = err.decode('utf-8')
        out_msg = out.decode('utf-8')
        ex_expected = "----> True"
        self.assertIn(ex_expected, out_msg, msg=err_msg)


class TestNumbaOptLevel(TestCase):
    # Tests that the setting of NUMBA_OPT influences the "cheap" module pass.
    # Spot checks NUMBA_OPT={'max', '3', '0'}

    def check(self, expected, opt_value, raw_value):
        # local imports for state-safety
        from numba import config, njit

        # check opt value and its raw_value
        self.assertEqual(config.OPT, opt_value)
        self.assertEqual(config.OPT._raw_value, raw_value)

        # Patch the CPUCodegen to make capture calls to the
        # `_module_pass_manager` through a `side_effect` function that asserts
        # that the kwargs being passed are as expected per the "NUMBA_OPT"
        # level. The `side_effect` function immediately raises with a knwon
        # message to abort further stages compilation once the check is
        # complete.
        from numba.core.codegen import CPUCodegen
        side_effect_message = "expected side effect"

        def side_effect(*args, **kwargs):
            self.assertEqual(kwargs, expected)
            raise RuntimeError(side_effect_message)

        with mock.patch.object(CPUCodegen, '_module_pass_manager',
                               side_effect=side_effect):
            with self.assertRaises(RuntimeError) as raises:
                njit(lambda : ...)()

            self.assertIn(side_effect_message, str(raises.exception))

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': 'max'})
    def test_opt_max(self):
        # NUMBA_OPT='max' should set opt to 3 and enable loop_vectorize
        expected = {'loop_vectorize': True,
                    'slp_vectorize': False,
                    'opt': 3,
                    'cost': 'cheap',
                    'vector_library': None}
        self.check(expected, 3, 'max')

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': '3'})
    def test_opt_3(self):
        # NUMBA_OPT='3' should not impact opt or loop_vectorize
        expected = {'loop_vectorize': False,
                    'slp_vectorize': False,
                    'opt': 0,
                    'cost': 'cheap',
                    'vector_library': None}
        self.check(expected, 3, 3)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': '0'})
    def test_opt_0(self):
        # NUMBA_OPT='0' should not impact opt or loop_vectorize
        expected = {'loop_vectorize': False,
                    'slp_vectorize': False,
                    'opt': 0,
                    'cost': 'cheap',
                    'vector_library': None}
        self.check(expected, 0, 0)

    @TestCase.run_test_in_subprocess()
    def test_opt_default(self):
        # NUMBA_OPT is not set, the default should not impact opt or
        # loop_vectorize
        expected = {'loop_vectorize': False,
                    'slp_vectorize': False,
                    'opt': 0,
                    'cost': 'cheap',
                    'vector_library': None}
        self.check(expected, 3, 3)

    @TestCase.run_test_in_subprocess(envvars={'NUMBA_OPT': 'invalid'})
    def test_opt_invalid(self):
        # NUMBA_OPT='invalid' should just proceed as default case
        expected = {'loop_vectorize': False,
                    'slp_vectorize': False,
                    'opt': 0,
                    'cost': 'cheap',
                    'vector_library': None}
        self.check(expected, 3, 3)


class TestVectorLibrary(TestCase):
    # Vector math libraries (e.g. Apple Accelerate, libmvec, Intel SVML) let
    # LLVM replace scalar math calls in vectorized loops with SIMD math calls.
    # At runtime, vector_library.resolve() maps a NUMBA_VECLIB setting to
    # (llvm_library_name, shared_lib_path): it locates the OS shared library,
    # checks via ctypes that the SIMD symbols LLVM expects are present, and
    # loads the library into LLVM's JIT runtime so compiled code can link them.

    def setUp(self):
        super().setUp()
        # clear that cache around each test so mocked libraries and host
        # libraries stay isolated.
        vector_library._load.cache_clear()
        self.addCleanup(vector_library._load.cache_clear)

    def test_candidates(self):
        cases = (
            ('darwin', 'arm64', ('darwin_libsystem_m', 'accelerate')),
            ('darwin', 'x86_64', ()),
            ('linux', 'x86_64', ()),
            ('linux2', 'AMD64', ()),
            ('win32', 'AMD64', ()),
            ('win32', 'x86_64', ()),
            ('darwin', 'amd64', ()),
            ('linux', 'aarch64', ()),
            ('freebsd', 'x86_64', ()),
        )
        for system, machine, expected in cases:
            with self.subTest(system=system, machine=machine):
                with (mock.patch.object(vector_library.sys, 'platform', system),
                      mock.patch.object(vector_library.platform, 'machine',
                                        return_value=machine)):
                    self.assertEqual(vector_library._candidates(), expected)

    def test_llvm_library_name(self):
        name = 'darwin_libsystem_m'
        with (mock.patch.object(vector_library, '_candidates',
                                return_value=(name,)),
              mock.patch.object(vector_library, '_load',
                                return_value='/mock/System') as load):
            self.assertEqual(vector_library.resolve(name),
                             (name, '/mock/System'))
            self.assertEqual(vector_library.resolve('auto'),
                             (name, '/mock/System'))
            self.assertEqual(load.call_count, 2)
        with self.assertRaises(ValueError):
            vector_library.resolve('libsystem')

    def test_config_and_validation(self):
        # Setting the NUMBA_VECLIB environment variable updates config.VECLIB.
        with override_env_config('NUMBA_VECLIB', 'auto'):
            self.assertEqual(config.VECLIB, 'auto')

        # 'none' disables vector math libraries: no OS library is searched for
        # or loaded into LLVM, and resolve() should return ('none', None).
        self.assertEqual(vector_library.resolve('none'), ('none', None))

        # Non-string or unrecognized NUMBA_VECLIB names fail with ValueError.
        for bad in (None, 123, 'invalid'):
            with self.assertRaisesRegex(ValueError,
                                        "Invalid NUMBA_VECLIB value"):
                vector_library.resolve(bad)

        # Every supported vector library must declare the SIMD C symbol names
        # that LLVM's vector-library pass emits, so _load() can verify them.
        for name in vector_library._LIBRARIES:
            self.assertTrue(vector_library._symbols(name))

    def test_resolve_fallback_and_errors(self):
        # Borrow three registered keys from _LIBRARIES so resolve() accepts
        # them without hardcoding library names, and simulate:
        # - missing: shared library is not installed (find_library -> None)
        # - invalid: shared library opens, but lacks expected SIMD symbols
        #            (bare object() makes getattr(lib, sym) raise)
        # - valid:   shared library opens and exposes all expected symbols
        valid, invalid, missing = tuple(vector_library._LIBRARIES)[:3]
        paths = {
            vector_library._LIBRARIES[invalid]: '/mock/bad.so',
            vector_library._LIBRARIES[valid]: '/mock/ok.so',
        }

        def mock_cdll(path, **_):
            return object() if path == '/mock/bad.so' else mock.MagicMock()

        # On an OS/CPU with no supported vector libraries, 'auto' gracefully
        # falls back to ('none', None), whereas asking for a specific library
        # raises RuntimeError because the user explicitly requested it.
        with mock.patch.object(vector_library, '_candidates', return_value=()):
            self.assertEqual(vector_library.resolve('auto'), ('none', None))
            with self.assertRaises(RuntimeError):
                vector_library.resolve(valid)

        with (
            mock.patch.object(vector_library, '_candidates',
                              return_value=(missing, invalid, valid)),
            mock.patch.object(vector_library, 'find_library',
                              side_effect=paths.get),
            mock.patch.object(vector_library.ctypes, 'CDLL',
                              side_effect=mock_cdll),
            mock.patch.object(vector_library.llvm,
                              'load_library_permanently') as mock_perm,
        ):
            # Explicitly requesting a missing shared library raises RuntimeError
            with self.assertRaisesRegex(RuntimeError, "could not be found"):
                vector_library.resolve(missing)

            # Explicitly requesting a library without the expected SIMD symbols
            # raises RuntimeError rather than letting LLVM fail at JIT link time
            with self.assertRaisesRegex(RuntimeError,
                                        "Cannot load vector library"):
                vector_library.resolve(invalid)

            # 'auto' tries candidates in order, skipping missing and invalid
            # libraries until it loads the first working one ('/mock/ok.so').
            self.assertEqual(vector_library.resolve('auto')[1], '/mock/ok.so')

            # Resolving that working library again reuses the cached result and
            # does not load the shared library into LLVM a second time.
            self.assertEqual(vector_library.resolve(valid)[1], '/mock/ok.so')
            mock_perm.assert_called_once_with('/mock/ok.so')


if __name__ == '__main__':
    unittest.main()
