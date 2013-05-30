# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import glob

modules_dir = os.path.join("source", "modules")
# mods = [os.path.splitext(fn)[0]
#             for fn in os.listdir(modules_dir) if fn.endswith(".rst")]
mods = [fn for fn in os.listdir(modules_dir) if fn.endswith(".rst")]
f = open(os.path.join(modules_dir, "modules.rst"), "w")

f.write("""
**********************
Numba Module Reference
**********************

Contents:

.. toctree::
   :titlesonly:
   :maxdepth: 2

""")

for mod in sorted(mods):
    f.write("   %s\n" % mod)
