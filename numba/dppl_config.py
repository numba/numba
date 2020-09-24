dppl_present = False

try:
    from dpctl.ocldrv import *
except:
    pass
else:
    dppl_present = True
