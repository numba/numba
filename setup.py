from distutils.core import setup, Extension
import numpy

setup(
    name = "numbapro",
    author = "Continuum Analytics, Inc.",
    author_email = "support@continuum.io",
    url = "http://www.continuum.io",
    license = "Proprietary",
    description = "compile Python code",
    ext_modules = [Extension(name = "numbapro._internal",
                             sources = ["src/_internal.c"],
                             include_dirs = [numpy.get_include()])],
    version = 0.5
)
