# Use Python.asdl to test bad ast.

import unittest, support, ast

class TestBad(support.SchemaTestCase):
    def test_module_missing_body(self):
        the_ast = ast.Module()
        with self.capture_error() as cap:
            self.schema.verify(the_ast)
        self.assertEqual(str(cap.exception),
                         "At Module.body: Missing field")

    def test_missing_expr_context(self):
        a_name = ast.Name(id='x') # missing expr_context
        a_binop = ast.BinOp(left=a_name, right =a_name, op=ast.Add())
        the_ast = ast.Expr(value=a_binop)
        with self.capture_error() as cap:
            self.schema.verify(the_ast)
        self.assertEqual(str(cap.exception),
                         "At Name.ctx: Missing field")

    def test_wrong_arg(self):
        bad = ast.Raise() # doesn't matter what is inside
        args = ast.arguments(args=[bad], defaults=[])
        the_ast = ast.FunctionDef(name="haha",
                                  args=args,
                                  body=[],
                                  decorator_list=[])
        with self.capture_error() as cap:
            self.schema.verify(the_ast)
        self.assertEqual(str(cap.exception),
                        "At arguments.args[0]: Cannot be a Raise")

    def test_return_missing_lineno(self):
        the_ast = ast.Return(col_offset=0)
        with self.capture_error() as cap:
            self.schema.verify(the_ast)
        self.assertEqual(str(cap.exception),
                         "At Return.lineno: Missing field")


if __name__ == '__main__':
    unittest.main()

