import os
from fnmatch import fnmatchcase
from distutils.util import convert_path
from distutils.core import setup
from Cython.Distutils import build_ext

from setupconfig import get_extensions

def find_packages(where='.', exclude=()):
    out = []
    stack=[(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where,name)
            if ('.' not in name and os.path.isdir(fn) and
                os.path.isfile(os.path.join(fn, '__init__.py'))
            ):
                out.append(prefix+name)
                stack.append((fn, prefix+name+'.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out

root = os.path.dirname(os.path.abspath(__file__))
setup(cmdclass={'build_ext': build_ext},
      ext_modules=get_extensions(path_prefix=root),
      packages=find_packages())
