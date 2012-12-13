import unittest
import os
import schema
import contextlib
import sys

def build_schema():
    '''Build a schema from Python.asdl
    '''
    return schema.load('Python.asdl')

class SchemaTestCase(unittest.TestCase):
    '''A base class for test cases that use the Python.asdl
    '''
    schema = build_schema()

    def capture_error(self):
        return self.assertRaises(schema.SchemaError)

# add parent path to import schema & asdl
sys.path += [os.path.dirname(__file__), '..']
