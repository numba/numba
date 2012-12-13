# Uses Python.asdl to test against some python script
# and some manually built ast


import unittest, support
import ast, os, inspect

class TestGood(support.SchemaTestCase):
    def test_self(self):
        # Cannot use __file__ because it may get the .pyc file instead
        srcfile = inspect.getsourcefile(type(self))
        self._test_script(srcfile)

    def test_schema_dot_py(self):
        self._test_script('../schema.py')

    def test_return_optional_value(self):
        the_ast = ast.Return(lineno=0,
                             col_offset=0)
        self.schema.verify(the_ast)
        the_ast = ast.Return(value=None,
                             lineno=0,
                             col_offset=0)
        self.schema.verify(the_ast)
        the_ast = ast.Return(value=ast.Name(id='x',
                                            ctx=ast.Load(),
                                            lineno=0,
                                            col_offset=0),
                             lineno=0, col_offset=0)
        self.schema.verify(the_ast)  # should not raise

    def _test_script(self, path):
        if path.startswith('.'):
            path = os.path.join(os.path.dirname(__file__), path)
        the_script = open(path).read()
        the_ast = ast.parse(the_script)
        self.schema.verify(the_ast)  # should not raise

if __name__ == '__main__':
    unittest.main()
