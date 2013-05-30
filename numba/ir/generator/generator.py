# -*- coding: utf-8 -*-

"""
Generate IR utilities from ASDL schemas.
"""

from __future__ import print_function, division, absolute_import

import os
import types
import codecs
import textwrap

from numba import PY3
from numba.asdl import asdl
from numba.asdl import schema
from numba.asdl import processor
from numba.asdl.asdl import pyasdl

if PY3:
    import io
    BytesIO = io.BytesIO
else:
    import cStringIO
    BytesIO = cStringIO.StringIO

root = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------------------
# Entry Points
#------------------------------------------------------------------------

# TODO: This needs to be parametrized when this is externalized
asdl_import_path = [os.path.join(root, os.pardir)]
import_processor = processor.ImportProcessor(pyasdl, asdl_import_path)

def parse(schema_name, schema_def, asdl_processor=import_processor):
    """
    Parse a schema given a schema name and schema string (ASDL string).
    """
    asdl_tree = get_asdl(schema_name, schema_def, asdl_processor)
    schema_instance = schema.build_schema(asdl_tree)
    return asdl_tree, schema_instance

def generate(schema_name, schema_str, codegens, file_allocator):
    """
    schema: ASDL schema (str)
    """
    asdl_tree, schema_instance = parse(schema_name, schema_str)

    for codegen in codegens:
        emitter = codegen.make_code_emitter(file_allocator)
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

def get_asdl(schema_name, schema, asdl_processor=None):
    parser, loader = asdl.load(schema_name, schema, pyasdl,
                               asdl_processor=asdl_processor)
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

    def close(self):
        for file in self.allocated_files.itervalues():
            file.close()

        self.allocated_files.clear()

class DiskFileAllocator(FileAllocator):

    def open_sourcefile(self, name):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = os.path.join(self.output_dir, name)
        file = codecs.open(filename, 'w', encoding='UTF-8')
        self.allocated_files[name] = file
        return file

class MemoryFileAllocator(FileAllocator):

    def open_sourcefile(self, name):
        file = StreamWriter(BytesIO())
        self.allocated_files[name] = file
        return file

#------------------------------------------------------------------------
# Code Generator Interface
#------------------------------------------------------------------------

class CodeEmitter(object):

    def __init__(self, outfile):
        self.outfile = outfile

    def emit(self, s):
        self.outfile.write(unicode(s))


class Codegen(object):
    """
    Interface for code generators.
    """

    def __init__(self, out_filename):
        self.out_filename = out_filename

    def make_code_emitter(self, file_allocator):
        outfile = file_allocator.open_sourcefile(self.out_filename)
        return CodeEmitter(outfile)

    def generate(self, emitter, asdl_tree, schema_instance):
        """
        Generate code for the given asdl tree. The ASDL tree is accompanied
        by a corresponding schema.Schema, which is easier to deal with.
        """


class UtilityCodegen(Codegen):
    """
    Generate some utility code:

        UtilityCode("foo.py", "my python code")
    """

    def __init__(self, out_filename, utility_code):
        super(UtilityCodegen, self).__init__(out_filename)
        self.utility_code = utility_code

    def generate(self, emitter, asdl_tree, schema_instance):
        emitter.emit(self.utility_code)


class SimpleCodegen(Codegen):

    def generate(self, emitter, asdl_tree, schema):
        self.emit_preamble(emitter, schema)

        for rulename, rule in schema.dfns.iteritems():
            if rule.is_sum:
                self.emit_nonterminal(emitter, schema, rulename, rule)

        for rulename, rule in schema.dfns.iteritems():
            if rule.is_product:
                self.emit_product(emitter, schema, rulename, rule)

        for rulename, rule in schema.dfns.iteritems():
            if rule.is_sum:
                for sumtype in rule.fields:
                    self.emit_sum(emitter, schema, rulename, rule, sumtype)

    def emit_preamble(self, emitter, schema):
        pass

    def emit_nonterminal(self, emitter, schema, rulename, rule):
        pass

    def emit_product(self, emitter, schema, rulename, rule):
        pass

    def emit_sum(self, emitter, schema, rulename, rule, sumtype):
        pass

if __name__ == '__main__':
    schema_def = textwrap.dedent("""
        module MyModule version "0.1"
        {
            mod = Module(object leaf1, object leaf2)
                | Foo(object leaf)

            foo = Add | Mul
            expr = X | Y
                 attributes (int lineno)

            alias = (int foo, int bar)
        }
    """)
    asdl_tree = get_asdl("MyASDL.asdl", schema_def)
    print("asdl", asdl_tree)
    print ("-------------")
    s = schema.build_schema(asdl_tree)
    print("dfns", s.dfns)
    print("types", s.types)
