# -*- coding: utf-8 -*-

"""
ASDL processor, handles imports. This should be part of the parser,
but we have many multiple copies of those...
"""

from __future__ import print_function, division, absolute_import

import os
import re
import tokenize

from numba.asdl.asdl import ASDLProcessor

#------------------------------------------------------------------------
# Process ASDL imports
#------------------------------------------------------------------------

class ImportProcessor(ASDLProcessor):

    def __init__(self, asdlmod, import_path):
        self.asdlmod = asdlmod
        self.import_path = import_path

    def preprocess(self, asdl_source):
        # Find and save import statements. Remove imports from source
        self.imports, source = find_imports(asdl_source)
        return source

    def postprocess(self, asdl_tree):
        # Import subtrees and types
        subtrees, types = apply_imports(self.asdlmod,
                                        self.imports,
                                        self.import_path)

        # Merge imported subtrees and types
        asdl_tree.dfns.extend(subtrees)
        asdl_tree.types.update(types)

        return asdl_tree

#------------------------------------------------------------------------
# Find Source Imports
#------------------------------------------------------------------------

pattern = "^\s*from (\w+) import (\*|\w+(?:, \w+)*)$"

def find_imports(asdl_source):
    """
    Find imports in the given ASDL source.

    :return: Two-tuple of (imports, source) where source has the imports
             removed and imports is a list of two-tuples (modname, (names))

             ([(str, (str,))], str)
    """
    lines = asdl_source.splitlines()
    imports = []

    source_lines = []
    for line in lines:
        m = re.match(pattern, line)
        if m is not None:
            module_name, names = m.groups()
            if names == '*':
                import_tuple = (module_name, (names,))
            else:
                import_tuple = (module_name, tuple(names.split(", ")))

            imports.append(import_tuple)
        else:
            source_lines.append(line)

    return imports, "\n".join(source_lines)

#------------------------------------------------------------------------
# Find Imported Terms
#------------------------------------------------------------------------

def apply_imports(asdlmod, imports, import_path):
    """
    Import ASDL subtrees from another schema along the given import path.
    """
    from . import asdl

    subtrees = []
    types = {}

    for modname, import_names in imports:
        # Find module file
        for path in import_path:
            fname = os.path.join(path, modname + '.asdl')
            if os.path.exists(fname):
                # Load ASDL tree
                parser, loader = asdl.load(modname, open(fname).read(), asdlmod)
                tree = loader.load()

                # Collect subtrees and types by name
                handle_import(tree, import_names, subtrees, types)
                break
        else:
            raise ImportError("No module named %r" % (modname + '.asdl',))

    return subtrees, types

def handle_import(tree, import_names, subtrees, types):
    for import_name in import_names:
        dfns = from_import(tree, import_name)
        subtrees.extend(dfns)
        for dfn in dfns:
            print(dfn.name, tree.types.keys())
            types[dfn.name] = tree.types[str(dfn.name)]

def from_import(tree, import_name):
    if import_name == '*':
        return tree.dfns #[dfn for dfn in tree.dfns if dfn.name in tree.types]

    for definition in tree.dfns:
        if str(definition.name) == import_name:
            return [definition]

    raise ImportError("Module %r has no rule %r" % (tree.name, import_name))
