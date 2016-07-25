from setuptools import setup

from source_module import cc

from numba.pycc.platform import _patch_exec_command


def run_setup():
    # Avoid sporadic crashes on Windows due to MSVCRT spawnve()
    _patch_exec_command()
    setup(ext_modules=[cc.distutils_extension()])


if __name__ == '__main__':
    run_setup()
