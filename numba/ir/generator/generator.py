# -*- coding: utf-8 -*-

"""
Generate IR utilities from ASDL schemas.
"""

from __future__ import print_function, division, absolute_import

import os
import types
import codecs
import textwrap
import cStringIO

from numba.asdl import asdl
from numba.asdl import schema
from numba.asdl.asdl import pyasdl

#------------------------------------------------------------------------
# Entry Points
#------------------------------------------------------------------------

def parse(schema_name, schema_def):
    """
    Parse a schema given a schema name and schema string (ASDL string).
    """
    asdl_tree = get_asdl(schema_name, schema_def)
    schema_instance = schema.build_schema(asdl_tree)
    return asdl_tree, schema_instance

def generate(schema_name, schema_str, codegens, file_allocator):
    """
    schema: ASDL schema (str)
    """
    asdl_tree, schema_instance = parse(schema_name, schema_str)

    for codegen in codegens:
        outfile = file_allocator.open_sourcefile(codegen.out_filename)
        emitter = CodeEmitter(outfile)
        codegen.generate(emitter, asdl_tree, schema_instance)

def generate_from_file(schema_filename, codegens, output_dir):
    """
    Generate code files for the given schema, code generators and output
    directory.

    Returns a file allocator with the open disk files.
    """
    schema_name = os.path.basename(schema_filename)
    schema_str = open(schema_filename).read()
    file_allocator = DiskFileAllocator(output_dir)
    generate(schema_name, schema_str, codegens, file_allocator)
    return file_allocator

def generate_module(file_allocator, name):
    """
    Generate an in-memory module from a generated Python implementation.
    """
    assert name in file_allocator.allocated_files

    f = file_allocator.allocated_files[name]
    f.seek(0)
    data = f.read()

    modname, _ = os.path.splitext(name)

    d = {}
    eval(compile(data, name, "exec"), d, d)
    m = types.ModuleType(modname)
    vars(m).update(d)

    return m

#------------------------------------------------------------------------
# Code Generator Utilities
#------------------------------------------------------------------------

def get_asdl(schema_name, schema):
    parser, loader = asdl.load(schema_name, schema, pyasdl)
    asdl_tree = loader.load()
    return asdl_tree

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

class CodeEmitter(object):

    def __init__(self, outfile):
        self.outfile = outfile

    def emit(self, s):
        self.outfile.write(str(s))


class Codegen(object):
    """
    Interface for code generators.
    """

    def __init__(self, out_filename):
        self.out_filename = out_filename

    def generate(self, emitter, asdl_tree, schema_instance):
        """
        Generate code for the given asdl tree. The ASDL tree is accompanied
        by a corresponding schema.Schema, which is easier to deal with.
        """

class UtilityCodeGen(Codegen):

    def __init__(self, out_filename, utility_code):
        super(UtilityCodeGen, self).__init__(out_filename)
        self.utility_code = utility_code

    def generate(self, emitter, asdl_tree, schema_instance):
        emitter.emit(self.utility_code)


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
