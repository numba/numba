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

        i32 = lc.Type.int(32)
        llvm_fnty = lc.Type.function(i32, [i32])
        module = lc.Module.new("test_module")
        function = module.get_or_insert_function(llvm_fnty, name='test_fn')
        assert function.is_declaration
        entry_block = function.append_basic_block('entry')
        builder = lc.Builder.new(entry_block)
        
        sig = typing.signature(types.int32, types.int32)
        typing_context = typing.Context()
        context = cpu.CPUContext(typing_context)

        # Ensure the cache is empty to begin with
        self.assertEqual(0, len(context.cached_internal_func))
        
        # After one compile, it should contain one entry
        context.compile_internal(builder, times2, sig, function.args)
        self.assertEqual(1, len(context.cached_internal_func))

        # After a second compilation of the same thing, it should still contain
        # one entry
        context.compile_internal(builder, times2, sig, function.args)
        self.assertEqual(1, len(context.cached_internal_func))

        # After compilation of another function, the cache should have grown by
        # one more.
        context.compile_internal(builder, times3, sig, function.args)
        self.assertEqual(2, len(context.cached_internal_func))

        sig2 = typing.signature(types.float64, types.float64)
        f64 = lc.Type.double()
        llvm_fnty2 = lc.Type.function(f64, [f64])
        function2 = module.get_or_insert_function(llvm_fnty2, name='test_fn_2')
        assert function2.is_declaration
        entry_block2 = function2.append_basic_block('entry')
        builder2 = lc.Builder.new(entry_block2)
        
        # Ensure that the same function with a different signature does not
        # reuse an entry from the cache in error
        context.compile_internal(builder2, times3, sig2, function2.args)
        self.assertEqual(3, len(context.cached_internal_func))


if __name__ == '__main__':
    unittest.main()

