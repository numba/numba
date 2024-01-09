import unittest
from numba import  types
from numba.core import ir, cgutils

# Import the record_is function
from np.arrayobj import record_is  # Replace with the actual module name

def create_mock_context():
    return cgutils.lowering.Context("test_module")

def create_mock_builder(context, func):
    block = context.append_basic_block(func, "entry")
    return cgutils.Builder(context, block)

class TestRecordIs(unittest.TestCase):

    def test_equal_records(self):
        sig = types.signature(types.boolean,
                              types.Record(('a', types.int32), ('b', types.int32)),
                              types.Record(('a', types.int32), ('b', types.int32)))

        context = create_mock_context()
        func = ir.Function(ir.Module(), ir.FunctionType(ir.types.Boolean(), [ir.types.Int(32), ir.types.Int(32)]), "test_function")
        builder = create_mock_builder(context, func)

        args = (cgutils.create_record(context, builder, sig.args[0]),
                cgutils.create_record(context, builder, sig.args[1]))

        result = record_is(context, builder, sig, args)
        self.assertTrue(result, "Expected True for equal records")

    def test_different_records(self):
        sig = types.signature(types.boolean,
                              types.Record(('a', types.int32), ('b', types.int32)),
                              types.Record(('a', types.int32), ('c', types.int32)))

        context = create_mock_context()
        func = ir.Function(ir.Module(), ir.FunctionType(ir.types.Boolean(), [ir.types.Int(32), ir.types.Int(32)]), "test_function")
        builder = create_mock_builder(context, func)

        args = (cgutils.create_record(context, builder, sig.args[0]),
                cgutils.create_record(context, builder, sig.args[1]))

        result = record_is(context, builder, sig, args)
        self.assertFalse(result, "Expected False for different records")

    def test_mixed_records(self):
        sig = types.signature(types.boolean,
                              types.Record(('a', types.int32), ('b', types.int32)),
                              types.Record(('a', types.int32), ('b', types.float64)))

        context = create_mock_context()
        func = ir.Function(ir.Module(), ir.FunctionType(ir.types.Boolean(), [ir.types.Int(32), ir.types.Int(32)]), "test_function")
        builder = create_mock_builder(context, func)

        args = (cgutils.create_record(context, builder, sig.args[0]),
                cgutils.create_record(context, builder, sig.args[1]))

        result = record_is(context, builder, sig, args)
        self.assertFalse(result, "Expected False for mixed records")

if __name__ == '__main__':
    unittest.main()
