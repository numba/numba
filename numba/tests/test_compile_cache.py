import llvm.core as lc
import numba.unittest_support as unittest
import numpy as np
from numba import types, typing
from numba.targets import cpu

class TestCompileCache(unittest.TestCase):

    def test_cache(self):
        def times2(i):
            return 2*i

        def times3(i):
            return i*3

        i32 = lc.Type.int(32)
        llvm_fnty = lc.Type.function(i32, [i32])
        module = lc.Module.new("test_module")
        function = module.get_or_insert_function(llvm_fnty, name='test_function')
        assert function.is_declaration
        entry_block = function.append_basic_block('entry')
        builder = lc.Builder.new(entry_block)
        
        sig = typing.signature(types.int32, types.int32)
        typing_context = typing.Context()
        context = cpu.CPUContext(typing_context).localized()

        # Ensure the cache is empty to begin with
        self.assertEqual(0, len(list(context.cached_internal_func.keys())))
        
        # After one compile, it should contain one entry
        context.compile_internal(builder, times2, sig, function.args)
        self.assertEqual(1, len(list(context.cached_internal_func.keys())))

        # After a second compilation of the same thing, it should still contain
        # one entry
        context.compile_internal(builder, times2, sig, function.args)
        self.assertEqual(1, len(list(context.cached_internal_func.keys())))

        # After compilation of another function, the cache should have grown by
        # one more.
        context.compile_internal(builder, times3, sig, function.args)
        self.assertEqual(2, len(list(context.cached_internal_func.keys())))

if __name__ == '__main__':
    unittest.main()

