# -*- coding: utf-8 -*-

"""
Generate Python AST nodes and Cython pxd files.
"""

from __future__ import print_function, division, absolute_import

import os

from numba.ir.generator import generator
from functools import partial

root = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(root, "tests")

#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class PythonASTNodeCodeGen(generator.Codegen):
    """
    Generate Python AST nodes.
    """

    def generate(self, emit_code, asdl_tree, schema):
        out = self.open_sourcefile("nodes.py")

        emit = partial(emit_code, out)
        emit_line = lambda s: emit(s + "\n")

        emit_line("class AST(object): pass")

        for rulename, rule in schema.dfns.iteritems():
            emit_line("class %s(AST): pass" % (rulename,))
            # print(rulename, rule.fields)
            if rule.is_sum:
                for subtype in rule.fields:
                    emit_line("class %s(%s): pass" % (subtype, rulename))


if __name__ == '__main__':
    schema_file = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_file, [PythonASTNodeCodeGen], root)
