import numba
from .support import TestCase, unittest
from numba import compiler, jitclass, ir
from numba.targets.registry import cpu_target
from numba.compiler import Pipeline, Flags, _PipelineManager
from numba.targets import registry
from numba import types, ir_utils, bytecode


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
        matched_call = ir_utils.find_callname(
            test_ir, test_ir.blocks[0].body[14].value, typemap)
        self.assertTrue(isinstance(matched_call, tuple) and
                        len(matched_call) == 2 and
                        matched_call[0] == 'append')

    def test_dead_code_elimination(self):

        class Tester(Pipeline):

            @classmethod
            def mk_pipeline(cls, args, return_type=None, flags=None, locals={},
                            library=None, typing_context=None,
                            target_context=None):
                if not flags:
                    flags = Flags()
                flags.nrt = True
                if typing_context is None:
                    typing_context = registry.cpu_target.typing_context
                if target_context is None:
                    target_context = registry.cpu_target.target_context
                return cls(typing_context, target_context, library, args,
                           return_type, flags, locals)

            def rm_dead_stage(self):
                ir_utils.dead_code_elimination(
                    self.func_ir, typemap=self.typemap)

            def compile_to_ir(self, func, DCE=False):
                """
                Compile and return IR
                """
                self.func_id = bytecode.FunctionIdentity.from_function(func)
                self.bc = self.extract_bytecode(self.func_id)
                self.lifted = []

                pm = _PipelineManager()
                pm.create_pipeline("pipeline")
                self.add_preprocessing_stage(pm)
                self.add_pre_typing_stage(pm)
                self.add_typing_stage(pm)
                if DCE is True:
                    pm.add_stage(self.rm_dead_stage, "DCE after typing")
                pm.finalize()
                pm.run(self.status)
                return self.func_ir

        def check_initial_ir(the_ir):
            # dead stuff:
            # a const int value 0xdead
            # an assign of above into to variable `dead`
            # del of both of the above
            # a const int above 0xdeaddead
            # an assign of said into to variable `deaddead`
            # del of both of the above
            # this is 8 things to remove

            self.assertEqual(len(the_ir.blocks), 1)
            block = the_ir.blocks[0]
            deads = []
            dels = [x for x in block.find_insts(ir.Del)]
            for x in block.find_insts(ir.Assign):
                if isinstance(getattr(x, 'target', None), ir.Var):
                    if 'dead' in getattr(x.target, 'name', ''):
                        deads.append(x)

            expect_removed = []
            self.assertEqual(len(deads), 2)
            expect_removed.extend(deads)
            del_names = [x.value for x in dels]
            for d in deads:
                # check the ir.Const is the definition and the value is expected
                const_val = the_ir.get_definition(d.value)
                self.assertTrue(int('0x%s' % d.target.name, 16),
                                const_val.value)
                expect_removed.append(const_val)

                # check there is a del for both sides of the assignment, one for
                # the dead variable and one for which to it the const gets
                # assigned
                self.assertIn(d.value.name, del_names)
                self.assertIn(d.target.name, del_names)

                for x in dels:
                    if x.value in (d.value.name, d.target.name):
                        expect_removed.append(x)
            self.assertEqual(len(expect_removed), 8)
            return expect_removed

        def check_dce_ir(the_ir):
            self.assertEqual(len(the_ir.blocks), 1)
            block = the_ir.blocks[0]
            deads = []
            consts = []
            dels = [x for x in block.find_insts(ir.Del)]
            for x in block.find_insts(ir.Assign):
                if isinstance(getattr(x, 'target', None), ir.Var):
                    if 'dead' in getattr(x.target, 'name', ''):
                        deads.append(x)
                if isinstance(getattr(x, 'value', None), ir.Const):
                    consts.append(x)
            self.assertEqual(len(deads), 0)
            # check there's no mention of dead in dels
            self.assertTrue(all(['dead' not in x.value for x in dels]))

            # check the consts to make sure there's no reference to 0xdead or
            # 0xdeaddead
            for x in consts:
                self.assertTrue(x.value.value not in [0xdead, 0xdeaddead])

        def foo(x):
            y = x + 1
            dead = 0xdead  # noqa
            z = y + 2
            deaddead = 0xdeaddead  # noqa
            ret = z * z
            return ret

        test_pipeline = Tester.mk_pipeline((types.intp,))
        no_dce = test_pipeline.compile_to_ir(foo)
        removed = check_initial_ir(no_dce)

        test_pipeline = Tester.mk_pipeline((types.intp,))
        w_dce = test_pipeline.compile_to_ir(foo, DCE=True)
        check_dce_ir(w_dce)

        # check that the count of initial - removed = dce
        self.assertEqual(len(no_dce.blocks[0].body) - len(removed),
                         len(w_dce.blocks[0].body))


if __name__ == "__main__":
    unittest.main()
