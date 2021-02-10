#!/usr/bin/env python
import runpy
import os

if bool(os.environ.get('NUMBA_USE_TYPEGUARD')):
    # The typeguard import hook must be installed prior to importing numba.
    # Therefore, this cannot be part of the numba package.
    from typeguard.importhook import install_import_hook
    install_import_hook(packages=['numba'])

# ensure full tracebacks are available and no help messages appear in test mode
os.environ['NUMBA_DEVELOPER_MODE'] = '1'


if __name__ == "__main__":
    runpy.run_module('numba.runtests', run_name='__main__')
