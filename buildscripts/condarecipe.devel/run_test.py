import sys
import os
import numba

workspace = os.environ.get('WORKSPACE', '')    # jenkins environment variable

if workspace:
    xmloutput = os.path.join(workspace, 'test-reports')
else:
    xmloutput = None

print("xmloutput ", xmloutput )
if not numba.test(xmloutput=xmloutput):
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
