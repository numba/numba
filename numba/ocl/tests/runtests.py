from __future__ import print_function, division, absolute_import
import sys
from .ocldrv import runtests as ocldrv
from .oclpy import runtests as oclpy


def test():
    return ocldrv.test() and oclpy.test()


if __name__ == '__main__':
    sys.exit(0 if test() else 1)
