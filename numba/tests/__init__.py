
try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()
