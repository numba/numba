from __future__ import print_function, division, absolute_import
from numbapro.testsupport import runtests


def test(**kwargs):
    return runtests(__name__, kwargs)
