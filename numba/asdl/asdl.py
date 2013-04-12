# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys, os.path

_importlist = "VisitorBase parse check".split()

root = os.path.join(os.path.dirname(__file__))
common_path = os.path.join(root, 'common')

#------------------------------------------------------------------------
# Parse ASDL Schemas
#------------------------------------------------------------------------

class ASDLParser(object):
    """
    ASDL parser that accepts string inputs. Defers to the given ASDL module
    implementation.
    """

    def __init__(self, asdlmod):
        self.asdlmod = asdlmod

    def parse(self, buf):
        """
        Parse an ASDL string.
        """
        scanner = self.asdlmod.ASDLScanner()
        parser = self.asdlmod.ASDLParser()

        tokens = scanner.tokenize(buf)
        try:
            return parser.parse(tokens)
        except self.asdlmod.ASDLSyntaxError as err:
            raise ValueError("Error while parsing schema: %s" % (err,))
            # print(err)
            # lines = buf.split("\n")
            # print((lines[err.lineno - 1])) # lines starts at 0, files at 1

    def check(self, mod, schema_name):
        """
        Check the validity of an ASDL parse.
        """
        v = self.asdlmod.Check()
        v.visit(mod)

        for t in v.types:
            if t not in mod.types and not t in self.asdlmod.builtin_types:
                v.errors += 1
                uses = ", ".join(v.types[t])
                print(("Undefined type %s, used in %s" % (t, uses)))

        if v.errors:
            raise ValueError(
                "Errors found while checking ASDL schema %r: %s" % (
                                             schema_name, v.errors))


#------------------------------------------------------------------------
# Load ASDL Schemas
#------------------------------------------------------------------------

class ASDLLoader(object):
    """
    Load a schema given an ASDLParser and an ASDL schema as a string.
    """

    def __init__(self, parser, schema_str, schema_name):
        self.parser = parser
        self.schema_str = schema_str
        self.schema_name = schema_name

    def load(self):
        asdl = self.parser.parse(self.schema_str)
        self.parser.check(asdl, self.schema_name)
        return asdl


#------------------------------------------------------------------------
# Get ASDL implementation
#------------------------------------------------------------------------

def get_asdl_pydir():
    major, minor = sys.version_info[0], sys.version_info[1]

    # Assumes that specific-path and common-path are a subdirectory
    # Build an absolute module path.
    prefix = __name__.rsplit('.', 1)

    # The else-case is for running tests in the current directory
    base = (prefix[0] + '.') if len(prefix) > 1 else ''
    dir = 'py%d_%d' % (major, minor)

    return base, dir


def _get_asdl_depending_on_version():
    """
    Return Python ASDL implementation depending on the Python version.
    """
    use_abs_import = 0

    base, dir = get_asdl_pydir()
    modname = base + dir + '.asdl'
    try:
        # try to import from version specific directory
        mod = __import__(modname, fromlist=_importlist, level=use_abs_import)
    except ImportError:
        # fallback to import from common directory
        dir = 'common'
        modname = base + dir + '.asdl'
        mod = __import__(modname, fromlist=_importlist)

    return mod


def load_pyschema(filename):
    """
    Load ASDL from the version-specific directory if the schema exists in
    there, otherwise from the 'common' subpackage.

    Returns a two-tuple (ASDLParser, ASDLLoader).
    """
    base, dir = get_asdl_pydir()
    version_specific_path = os.path.join(root, dir)

    srcfile = os.path.join(version_specific_path, filename)
    if not os.path.exists(srcfile):
        srcfile = os.path.join(common_path, filename)
        from numba.asdl.common import asdl as asdlmod
    else:
        asdlmod = _get_asdl_depending_on_version()

    asdl_str = open(srcfile).read()
    parser = ASDLParser(asdlmod)
    loader = ASDLLoader(parser, asdl_str, filename)
    return parser, loader

#------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------

python_parser, python_loader = load_pyschema("Python.asdl")

# Python version-specific parsed ASDL
python_asdl = python_loader.load()

# Python version-specific asdl implementation
pyasdl = python_parser.asdlmod
