# -*- coding: utf-8 -*-

"""
Generate IR utilities from ASDL schemas.
"""

from __future__ import print_function, division, absolute_import

import os
import codecs

from numba.asdl import asdl
from numba.asdl.asdl import pyasdl

def generate(schema, output_dir):
    """
    schema: ASDL schema (str)
    """
    pass

def generate_from_file(schema_filename, output_dir):
    schema_str = open(schema_filename).read()
    generate(schema_str, output_dir)

#------------------------------------------------------------------------
# Code Generator Interface
#------------------------------------------------------------------------

class Codegen(object):
    """
    Interface for code generators.
    """

    def __init__(self, schema_name, schema, output_dir):
        self.schema_name = schema_name
        self.schema = schema
        self.output_dir = output_dir

        parser, loader = asdl.load(schema_name, schema, pyasdl)
        self.asdl_tree = loader.load()

    def open_sourcefile(self, name):
        filename = os.path.join(self.output_dir, name)
        return codecs.open(filename, 'w', encoding='UTF-8')


#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class PythonASTNodeCodeGen(Codegen):
    """
    Generate Python AST nodes.
    """