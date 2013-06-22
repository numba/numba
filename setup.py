import os
from fnmatch import fnmatchcase
from distutils.core import setup, Extension
from distutils.util import convert_path

import versioneer

versioneer.versionfile_source = 'numbapro/_version.py'
versioneer.versionfile_build = 'numbapro/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'numbapro-'

ext_modules = [
    Extension(
        name = 'numbapro._utils.mviewbuf',
        sources = ['numbapro/_utils/mviewbuf.c'],
        # extra_compile_args = ['-pedantic', '-ansi'], # for checking C90
    ),
]

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


cmdclass = versioneer.get_cmdclass()

setup(
    name = "numbapro",
    version = versioneer.get_version(),
    author = "Continuum Analytics, Inc.",
    author_email = "support@continuum.io",
    url = "http://www.continuum.io",
    license = "Proprietary",
    description = "compile Python code",
    ext_modules = ext_modules,
    packages = find_packages(),
    cmdclass = cmdclass,
)

