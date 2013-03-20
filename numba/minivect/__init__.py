# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import os

root = os.path.dirname(os.path.abspath(__file__))

def get_include():
    return os.path.join(root, 'include')
