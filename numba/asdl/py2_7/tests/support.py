import unittest
import os
import asdl
from schema import SchemaBuilder, SchemaError
import contextlib

def build_schema():
    srcfile = os.path.join(os.path.dirname(__file__), '../Python.asdl')

    python_asdl = asdl.parse(srcfile)
    assert asdl.check(python_asdl)

    schblr = SchemaBuilder()
    schblr.visit(python_asdl)
    schema = schblr.schema
    return schema

class SchemaTestCase(unittest.TestCase):
    schema = build_schema()

    def capture_error(self):
        return self.assertRaises(SchemaError)

import sys
sys.path += [os.path.dirname(__file__), '..'] # add parent path
