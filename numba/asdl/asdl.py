import sys, os.path

_importlist = "VisitorBase parse check".split()

_cd = os.path.join(os.path.dirname(__file__))
version_specific_path = None
common_path = os.path.join(_cd, 'common')

def _get_asdl_depending_on_version():
    '''Export names in the correct asdl.py depending on the python version.
    '''

    global version_specific_path
    major, minor = sys.version_info[0], sys.version_info[1]
    # Assumes that specific-path and common-path are a subdirectory
    # Build an absolute module path.
    prefix = __name__.rsplit('.', 1)
    # The else-case is for running tests in the current directory
    base = (prefix[0] + '.') if len(prefix) > 1 else ''
    dir = 'py%d_%d' % (major, minor)
    version_specific_path = os.path.join(_cd, dir)

    use_abs_import = 0

    modname = base + dir + '.asdl'
    try:
        # try to import from version specific directory
        mod = __import__(modname, fromlist=_importlist, level=use_abs_import)
    except ImportError:
        # fallback to import from common directory
        dir = 'common'
        modname = base + dir + '.asdl'
        mod = __import__(modname, fromlist=_importlist) #, level=use_rel_and_abs)
    for i in _importlist:
        globals()[i] = getattr(mod, i)

def load(filename):
    '''Load ASDL from the version_specific_path if exists,
    or from the generic_path.
    '''
    srcfile = os.path.join(version_specific_path, filename)
    if not os.path.exists(srcfile):
        srcfile = os.path.join(common_path, filename)
    asdl = parse(srcfile)
    assert check(asdl), "Invalid asdl %s" % srcfile
    return asdl

# initialize
_get_asdl_depending_on_version()
