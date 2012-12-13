import unittest, support
import ast, os

class TestGood(support.SchemaTestCase):
    def test_self(self):
        self._test_script(__file__)

    def test_schema_dot_py(self):
        self._test_script('../schema.py')

    def test_return_optional_value(self):
        the_ast = ast.Return()
        self.schema.verify(the_ast)
        the_ast = ast.Return(value=None)
        self.schema.verify(the_ast)
        the_ast = ast.Return(value=ast.Name(id='x', ctx=ast.Load()))
        self.schema.verify(the_ast)

    def _test_script(self, path):
        if path.startswith('.'):
            path = os.path.join(os.path.dirname(__file__), path)
        the_script = open(path).read()
        the_ast = ast.parse(the_script)
        self.schema.verify(the_ast) # should not raise

if __name__ == '__main__':
    unittest.main()
