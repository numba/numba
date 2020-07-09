dppy_present = False

try:
    from dppy.ocldrv import *
except:
    pass
else:
    dppy_present = True
