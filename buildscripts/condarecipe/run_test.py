import sys
import numbapro
if not numbapro.test():
    sys.exit(1)
print('numbapro.__version__: %s' % numbapro.__version__)
