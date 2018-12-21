import numba
from numba import unittest_support as unittest
from .support import TestCase
from numba import compiler, jitclass
from numba.targets.registry import cpu_target


@jitclass([('val', numba.types.List(numba.intp))])
class Dummy(object):
    def __init__(self, val):
        self.val = val


class TestIrUtils(TestCase):
    """
    Tests ir handling utility functions like find_callname.
    """

    def test_obj_func_match(self):
        """Test matching of an object method (other than Array see #3449)
        """

        def test_func():
            d = Dummy([1])
            d.val.append(2)

        test_ir = compiler.run_frontend(test_func)
        typingctx = cpu_target.typing_context
        typemap, _, _ = compiler.type_inference_stage(
            typingctx, test_ir, (), None)
        matched_call = numba.ir_utils.find_callname(
            test_ir, test_ir.blocks[0].body[14].value, typemap)
        self.assertTrue(isinstance(matched_call, tuple)
                        and len(matched_call) == 2
                        and matched_call[0] == 'append')


if __name__ == "__main__":
    unittest.main()
