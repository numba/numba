# -*- coding: utf-8 -*-

"""
Generate IR utilities from ASDL schemas.
"""

from __future__ import print_function, division, absolute_import

import os
import codecs
import textwrap
import cStringIO

from numba.asdl import asdl
from numba.asdl import schema
from numba.asdl.asdl import pyasdl

#------------------------------------------------------------------------
# Entry Points
#------------------------------------------------------------------------

def generate(schema_name, schema_def, codegen_classes, file_allocator):
    """
    schema: ASDL schema (str)
    """
    asdl_tree = get_asdl(schema_name, schema_def)
    schema_instance = schema.build_schema(asdl_tree)

    print(schema_instance)

    for codegen_class in codegen_classes:
        codegen = codegen_class(schema_name, schema_def, file_allocator)
        codegen.generate(emit_code, asdl_tree, schema_instance)

def generate_from_file(schema_filename, codegens, output_dir):
    schema_name = os.path.basename(schema_filename)
    schema_str = open(schema_filename).read()
    generate(schema_name, schema_str, codegens, DiskFileAllocator(output_dir))

#------------------------------------------------------------------------
# Code Generator Utilities
#------------------------------------------------------------------------

def get_asdl(schema_name, schema):
    parser, loader = asdl.load(schema_name, schema, pyasdl)
    asdl_tree = loader.load()
    return asdl_tree

def emit_code(file, obj):
    file.write(str(obj))

def is_simple(sum):
    """
    Return True if a sum is a simple.

    A sum is simple if its types have no fields, e.g.
    unaryop = Invert | Not | UAdd | USub
    """
    for t in sum.types:
        if t.fields:
            return False
    return True

#------------------------------------------------------------------------
# File Handling
#------------------------------------------------------------------------

StreamWriter = codecs.getwriter('UTF-8')

class FileAllocator(object):

    def __init__(self, output_dir=None):
        self.output_dir = output_dir

        # file_name -> file
        self.allocated_files = {}

    def open_sourcefile(self, name):
        "Allocate a file and save in in allocated_files"

class DiskFileAllocator(FileAllocator):

    def open_sourcefile(self, name):
        filename = os.path.join(self.output_dir, name)
        file = codecs.open(filename, 'w', encoding='UTF-8')
        self.allocated_files[name] = file
        return file

class MemoryFileAllocator(FileAllocator):

    def open_sourcefile(self, name):
        file = StreamWriter(cStringIO.StringIO())
        self.allocated_files[name] = file
        return file

#------------------------------------------------------------------------
# Code Generator Interface
#------------------------------------------------------------------------

class Codegen(object):
    """
    Interface for code generators.
    """

    def __init__(self, schema_name, schema, file_allocator):
        self.schema_name = schema_name
        self.schema = schema
        self.file_allocator = file_allocator

    def open_sourcefile(self, name):
        return self.file_allocator.open_sourcefile(name)

    def generate(self, emit_code, asdl_tree, schema_instance):
        """
        Generate code for the given asdl tree. The ASDL tree is accompanied
        by a corresponding schema.Schema, which is easier to deal with.
        """



if __name__ == '__main__':
    schema_def = textwrap.dedent("""
        module MyModule version "0.1"
        {
            mod = Module(object leaf1, object leaf2)
                | Foo(object leaf)
        }
    """)
    asdl_tree = get_asdl("MyASDL.asdl", schema_def)
    print(asdl_tree)
    s = schema.build_schema(asdl_tree)
    print(s)
