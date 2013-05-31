# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import sys
import shutil
import tempfile
import subprocess

from .. import naming, build

root = os.path.dirname(os.path.abspath(__file__))
schema_filename = os.path.join(root, "testschema1.asdl")

def filenames(codegens):
    return [c.out_filename for c in codegens]

all_features = [naming.interface + '.pxd', naming.nodes + '.pxd',
                naming.visitor + '.pxd', naming.transformer + '.pxd',
                naming.interface + '.py',  naming.nodes + '.py',
                naming.visitor + '.py', naming.transformer + '.py',]

def test_features():
    """
    >>> test_features()
    """
    codegens = build.enumerate_codegens(['ast'], mask=0)
    fns = filenames(codegens)
    assert set(fns) == set([naming.interface + '.pxd', naming.nodes + '.pxd',
                            naming.interface + '.py',  naming.nodes + '.py',])

    codegens = build.enumerate_codegens(['ast'], mask=build.cython)
    fns = filenames(codegens)
    assert set(fns) == set([naming.interface + '.py',  naming.nodes + '.py',])

    codegens = build.enumerate_codegens(
        ['ast', 'visitor', 'transformer'], mask=0)
    fns = filenames(codegens)
    assert set(fns) == set(all_features)

def test_package_building():
    """
    >>> test_package_building()
    """
    features = ['ast', 'visitor', 'transformer']
    output_dir = tempfile.mkdtemp()
    try:
        build.build_package(schema_filename, features, output_dir)
        for feature_file in all_features:
            assert os.path.exists(os.path.join(output_dir, feature_file))
    finally:
        shutil.rmtree(output_dir)

def test_package_compilation():
    """
    >>> test_package_compilation()
    """
    features = ['ast', 'visitor', 'transformer']
    output_dir = tempfile.mkdtemp()
    try:
        build.build_package(schema_filename, features, output_dir)
        p = subprocess.Popen([sys.executable, "setup.py",
                              "build_ext", "--inplace"], cwd=output_dir)
        assert p.wait() == 0, p.poll()
    finally:
        shutil.rmtree(output_dir)


if __name__ == '__main__':
    import doctest
    try:
        import cython
    except ImportError:
        print("Skipping test, cython not installed")
    else:
        sys.exit(0 if doctest.testmod().failed == 0 else 1)