import os
import functools
from distutils.extension import Extension

import numpy as np

def prefix_module(prefix, module_name):
    if prefix:
        return "%s.%s" % (prefix, module_name)

    return module_name

def prefix_path(prefix, path):
    if prefix:
        return "%s/%s" % (prefix.rstrip("/"), path.lstrip("/"))

    return path

def make_extension(path_prefix, module_prefix, modname, sources, depends, **kwds):
    _prefix_path = functools.partial(prefix_path, path_prefix)

    return Extension(
        prefix_module(module_prefix, modname),
        sources=list(map(_prefix_path, sources)),
        depends=list(map(_prefix_path, depends)),
        **kwds
    )

def get_extensions(path_prefix, module_prefix=""):
    include_dirs = [prefix_path(path_prefix, 'include'),
                    np.get_include()]

    perfecthash_deps = ["include/perfecthash.h"]

    Extension = functools.partial(make_extension, path_prefix, module_prefix)

    extensions = [
        Extension("extensibletype.extensibletype",
                  ["extensibletype/extensibletype.pyx",
                   #'../ulib/src/base/md5sum.c',
                   #'../ulib/src/base/hash.c'
                  ],
                  include_dirs=include_dirs,
                  depends=perfecthash_deps),

        Extension("extensibletype.intern",
                  ["extensibletype/intern.pyx"],
                  include_dirs=include_dirs,
                  depends=["include/globalinterning.h",
                           "include/interning.h",
                           "include/perfecthash.h"]),

        Extension("extensibletype.methodtable",
                  ["extensibletype/methodtable.pyx"],
                  include_dirs=include_dirs,
                  depends=perfecthash_deps),

        Extension("extensibletype.test.pstdint",
                  ["extensibletype/test/pstdint.pyx"],
                  include_dirs=include_dirs,
                  depends=["include/pstdint.h"]),
    ]

    return extensions
