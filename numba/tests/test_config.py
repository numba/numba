import os
import tempfile
import unittest
from .support import TestCase, temp_directory, override_env_config
from numba import config

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


if __name__ == '__main__':
    unittest.main()
