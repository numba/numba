from __future__ import print_function, absolute_import
from math import pi, e

global_dict = globals()

constants = {'pi':pi, 'e':e}

for name, op in constants.items():
    global_dict[name] = op
