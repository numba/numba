import re

def _sentry_llvm_version():
    """
    Make sure we meet min llvmpy version
    """
    import warnings
    import llvm
    min_version = (0, 12, 6)

    # Only look at the the major, minor and bugfix version numbers.
    # Ignore other stuffs
    regex = re.compile(r'(\d+)\.(\d+).(\d+)')
    m = regex.match(llvm.__version__)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver < min_version:
            msg = ("Numba requires at least version %d.%d.%d of llvmpy.\n"
                   "Installed version is %s.\n"
                   "Please update llvmpy." %
                   (min_version + (llvm.__version__,)))
            raise ImportError(msg)
    else:
        # Not matching?
        warnings.warn("llvmpy version format not recognized!")

