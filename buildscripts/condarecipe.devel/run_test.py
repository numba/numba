import sys
import os
import numba
import subprocess

subprocess.check_call("conda install --yes pip".split())
subprocess.check_call("pip install xmlrunner".split())

workspace = os.environ.get('WORKSPACE', '')    # jenkins environment variable

if workspace:
    xmloutput = os.path.join(workspace, 'test-reports')
else:
    xmloutput = None

try:
    import xmlrunner
except ImportError:
    # Disable xmloutput if xmlrunner is not available
    xmloutput = None

print("xmloutput ", xmloutput)

if not numba.test(xmloutput=xmloutput):
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
