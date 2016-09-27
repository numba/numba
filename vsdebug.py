import sys
import os.path
import runpy

target = sys.argv[1]
relpath = os.path.relpath(target)
root, ext = os.path.splitext(relpath)
modpath = root.replace(os.path.sep, '.')

del sys.argv[1]
runpy.run_module(modpath, run_name="__main__", alter_sys=True)