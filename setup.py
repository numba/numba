import os
from distutils.core import setup, Extension
from distutils.util import convert_path

import versioneer

versioneer.versionfile_source = 'numbapro/_version.py'
versioneer.versionfile_build = 'numbapro/_version.py'
versioneer.tag_prefix = ''
versioneer.parentdir_prefix = 'numbapro-'

ext_modules = [
    Extension(
        name='numbapro.vectorizers.workqueue',
        sources=['numbapro/vectorizers/workqueue.c'],
        depends=['numbapro/vectorizers/workqueue.h'],
    )
]


def find_packages(rootdir):
    out = []
    stack = [convert_path(rootdir)]
    while stack:
        where = stack.pop()
        for name in os.listdir(where):
            path = os.path.join(where, name)
            if os.path.isfile(path) and name == '__init__.py':
                out.append(where.replace(os.path.sep, '.'))
            elif os.path.isdir(path):
                stack.append(path)
    print(out)
    return out


cmdclass = versioneer.get_cmdclass()

setup(
    name="numbapro",
    version=versioneer.get_version(),
    author="Continuum Analytics, Inc.",
    author_email="support@continuum.io",
    url="http://www.continuum.io",
    license="Proprietary",
    description="compile Python code",
    ext_modules=ext_modules,
    packages=find_packages('numbapro'),
    cmdclass=cmdclass,
)

