dppl_present = False

try:
    from dppl.ocldrv import *
except:
    pass
else:
    dppl_present = True
