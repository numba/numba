import numba.unittest_support as unittest

import llvmlite.llvmpy.core as lc

import numpy as np

from numba import types, typing
from numba.targets import cpu


class TestCompileCache(unittest.TestCase):
    '''
    Tests that the caching in BaseContext.compile_internal() works correctly by
    checking the state of the cache when it is used by the CPUContext.
    '''

    def test_cache(self):
        def times2(i):
            return 2*i

        def times3(i):
            return i*3

        def make_closure(x, y):
            def f(z):
                return y + z
            return f

        typing_context = typing.Context()
        context = cpu.CPUContext(typing_context)
        module = lc.Module.new("test_module")

        sig = typing.signature(types.int32, types.int32)
        llvm_fnty = context.call_conv.get_function_type(sig.return_type,
                                                        sig.args)
        function = module.get_or_insert_function(llvm_fnty, name='test_fn')
        args = context.call_conv.get_arguments(function)
        assert function.is_declaration
        entry_block = function.append_basic_block('entry')
        builder = lc.Builder.new(entry_block)

        # Ensure the cache is empty to begin with
        self.assertEqual(0, len(context.cached_internal_func))
        
        # After one compile, it should contain one entry
        context.compile_internal(builder, times2, sig, args)
        self.assertEqual(1, len(context.cached_internal_func))

        # After a second compilation of the same thing, it should still contain
        # one entry
        context.compile_internal(builder, times2, sig, args)
        self.assertEqual(1, len(context.cached_internal_func))

        # After compilation of another function, the cache should have grown by
        # one more.
        context.compile_internal(builder, times3, sig, args)
        self.assertEqual(2, len(context.cached_internal_func))

        sig2 = typing.signature(types.float64, types.float64)
        llvm_fnty2 = context.call_conv.get_function_type(sig2.return_type,
                                                         sig2.args)
        function2 = module.get_or_insert_function(llvm_fnty2, name='test_fn_2')
        args2 = context.call_conv.get_arguments(function2)
        assert function2.is_declaration
        entry_block2 = function2.append_basic_block('entry')
        builder2 = lc.Builder.new(entry_block2)
        
        # Ensure that the same function with a different signature does not
        # reuse an entry from the cache in error
        context.compile_internal(builder2, times3, sig2, args2)
        self.assertEqual(3, len(context.cached_internal_func))

        # Closures with distinct cell contents must each be compiled.
        clo11 = make_closure(1, 1)
        clo12 = make_closure(1, 2)
        clo22 = make_closure(2, 2)
        res1 = context.compile_internal(builder, clo11, sig, args)
        self.assertEqual(4, len(context.cached_internal_func))
        res2 = context.compile_internal(builder, clo12, sig, args)
        self.assertEqual(5, len(context.cached_internal_func))
        # Same cell contents as above (first parameter isn't captured)
        res3 = context.compile_internal(builder, clo22, sig, args)
        self.assertEqual(5, len(context.cached_internal_func))


if __name__ == '__main__':
    unittest.main()

