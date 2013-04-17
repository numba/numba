# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import os
import types

from .. import generator

root = os.path.dirname(os.path.abspath(__file__))

def schema_filename(name):
    return os.path.join(root, name)

def load_schema(name):
    schema_fn = schema_filename(name)
    schema_name = os.path.basename(schema_fn)
    schema_str = open(schema_fn).read()
    return schema_name, schema_str

def generate_in_memory(schema_name, codegens):
    file_allocator = generator.MemoryFileAllocator()
    schema_name, schema_str = load_schema(schema_name)
    generator.generate(schema_name, schema_str, codegens, file_allocator)
    return file_allocator

def generate_module(file_allocator, name):
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