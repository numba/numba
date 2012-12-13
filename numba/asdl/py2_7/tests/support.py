import unittest
import os
import asdl
from schema import SchemaBuilder, SchemaError
import contextlib
import sys

def build_schema():
    '''Build a schema from Python.asdl
    '''
    srcfile = os.path.join(os.path.dirname(__file__), '../Python.asdl')

    python_asdl = asdl.parse(srcfile)
    assert asdl.check(python_asdl)

    schblr = SchemaBuilder()
    schblr.visit(python_asdl)
    schema = schblr.schema
    return schema

class SchemaTestCase(unittest.TestCase):
    '''A base class for test cases that use the Python.asdl
    '''
    schema = build_schema()

    def capture_error(self):
        return self.assertRaises(SchemaError)

# add parent path to import schema & asdl
sys.path += [os.path.dirname(__file__), '..']
