"""
Simple testing script for library finder

"""

from __future__ import print_function
import os
from numbapro.cudadrv import libs

basedir = '/Users/sklam/dev/cudatoolkit/5.5v1'
platforms = [
    ('linux32', 'linux2'),
    ('linux64', 'linux2'),
    ('osx', 'darwin'),
    ('win32', 'win32'),
    ('win64', 'win32'),
    ]

for dir, plat in platforms:
    libdir = os.path.join(basedir, dir)
    os.environ['NUMBAPRO_CUDALIB'] = libdir
    print(('Testing %s' % libdir).center(80, '='))
    assert libs.test(plat)

