import unittest
from numba import types, errors
from numba.templating.templates import AbstractTemplate, signature
from numba.core.typing.builtins import Float  
class TestFloatTemplate(unittest.TestCase):

    def test_float_template(self):
        float_template = Float()

        # Test case 1: Test with a float argument
        float_arg = types.float64
        result_signature = float_template.generic([float_arg], {})
        expected_signature = signature(float_arg, float_arg)
        self.assertEqual(result_signature, expected_signature)

        # Test case 2: Test with an integer argument
        int_arg = types.int32
        result_signature = float_template.generic([int_arg], {})
        expected_signature = signature(types.float64, int_arg)
        self.assertEqual(result_signature, expected_signature)

        # Test case 3: Test with a complex argument
        complex_arg = types.complex128
        with self.assertRaises(errors.NumbaTypeError):
            float_template.generic([complex_arg], {})

        # Test case 4: Test with a non-numeric argument
        non_numeric_arg = types.StringLiteral('abc')
        with self.assertRaises(errors.NumbaTypeError):
            float_template.generic([non_numeric_arg], {})

        # Test case 5: Test with float("inf")
        inf_arg = types.float64
        inf_arg_const = types.Const(float("inf"), inf_arg)
        result_signature = float_template.generic([inf_arg_const], {})
        expected_signature = signature(types.float64, inf_arg_const)
        self.assertEqual(result_signature, expected_signature)

if __name__ == '__main__':
    unittest.main()
