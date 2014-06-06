from math import pi, e

global_dict = globals()

constants = {'pi':pi, 'e':e}

for name, op in constants.items():
    global_dict[name] = op
