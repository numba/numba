from __future__ import print_function, absolute_import

from llvmlite import ir, binding as ll

from numba import types
from numba import unittest_support as unittest
from numba import datamodel


def test_factory(name, fetype, support_as_data=True):
    dmm = datamodel.defaultDataModelManager

    class DataModelTester(unittest.TestCase):
        datamodel = NotImplemented

        def setUp(self):
            self.module = ir.Module()

        def test_as_arg(self):
            fnty = ir.FunctionType(ir.VoidType(), [])
            function = ir.Function(self.module, fnty, name="test_as_arg")
            builder = ir.IRBuilder()
            builder.position_at_end(function.append_basic_block())

            undef_value = ir.Constant(self.datamodel.get_value_type(), None)
            args = self.datamodel.as_argument(builder, undef_value)
            self.assertIsNot(args, NotImplemented, "as_argument returned "
                                                   "NotImplementedError")

            if isinstance(args, (tuple, list)):
                def recur_flatten_type(args):
                    for arg in args:
                        if isinstance(arg, (tuple, list)):
                            yield tuple(recur_flatten_type(arg))
                        else:
                            yield arg.type

                def recur_tuplize(args):
                    for arg in args:
                        if isinstance(arg, (tuple, list)):
                            yield tuple(recur_tuplize(arg))
                        else:
                            yield arg

                argtypes = tuple(recur_flatten_type(args))
                exptypes = tuple(recur_tuplize(
                    self.datamodel.get_argument_type()))
                self.assertEqual(exptypes, argtypes)
            else:
                self.assertEqual(args.type,
                                 self.datamodel.get_argument_type())

            rev_value = self.datamodel.from_argument(builder, args)
            self.assertEqual(rev_value.type, self.datamodel.get_value_type())

            builder.ret_void()  # end function

            # Ensure valid LLVM generation
            materialized = ll.parse_assembly(str(self.module))
            print(materialized)

        def test_as_return(self):
            fnty = ir.FunctionType(ir.VoidType(), [])
            function = ir.Function(self.module, fnty, name="test_as_return")
            builder = ir.IRBuilder()
            builder.position_at_end(function.append_basic_block())

            undef_value = ir.Constant(self.datamodel.get_value_type(), None)
            ret = self.datamodel.as_return(builder, undef_value)
            self.assertIsNot(ret, NotImplemented, "as_return returned "
                                                  "NotImplementedError")

            self.assertEqual(ret.type, self.datamodel.get_return_type())

            rev_value = self.datamodel.from_return(builder, ret)
            self.assertEqual(rev_value.type, self.datamodel.get_value_type())

            builder.ret_void()  # end function

            # Ensure valid LLVM generation
            materialized = ll.parse_assembly(str(self.module))
            print(materialized)

        if support_as_data:
            def test_as_data(self):
                fnty = ir.FunctionType(ir.VoidType(), [])
                function = ir.Function(self.module, fnty, name="test_as_data")
                builder = ir.IRBuilder()
                builder.position_at_end(function.append_basic_block())

                undef_value = ir.Constant(self.datamodel.get_value_type(), None)
                data = self.datamodel.as_data(builder, undef_value)
                self.assertIsNot(data, NotImplemented, "as_data returned "
                                                       "NotImplementedError")

                self.assertEqual(data.type, self.datamodel.get_data_type())

                rev_value = self.datamodel.from_data(builder, data)
                self.assertEqual(rev_value.type,
                                 self.datamodel.get_value_type())

                builder.ret_void()  # end function

                # Ensure valid LLVM generation
                materialized = ll.parse_assembly(str(self.module))
                print(materialized)
        else:
            def test_as_data_not_supported(self):
                fnty = ir.FunctionType(ir.VoidType(), [])
                function = ir.Function(self.module, fnty, name="test_as_data")
                builder = ir.IRBuilder()
                builder.position_at_end(function.append_basic_block())

                undef_value = ir.Constant(self.datamodel.get_value_type(), None)
                data = self.datamodel.as_data(builder, undef_value)
                self.assertIs(data, NotImplemented)
                rev_data = self.datamodel.from_data(builder, undef_value)
                self.assertIs(rev_data, NotImplemented)

    model = dmm.lookup(fetype)
    testcls = type(name, (DataModelTester,), {'datamodel': model})
    glbls = globals()
    assert name not in glbls
    glbls[name] = testcls



test_factory("TestBool", types.boolean)
test_factory("TestPyObject", types.pyobject)


for bits in [8, 16, 32, 64]:
    # signed
    test_factory("TestInt{0}".format(bits),
                 getattr(types, 'int{0}'.format(bits)))
    # unsigned
    test_factory("TestUInt{0}".format(bits),
                 getattr(types, 'uint{0}'.format(bits)))

test_factory("TestFloat", types.float32)
test_factory("TestDouble", types.float64)
test_factory("TestComplex", types.complex64)
test_factory("TestDoubleComplex", types.complex128)

test_factory("TestPointerOfInt32", types.CPointer(types.int32))

test_factory("TestUniTupleOf2xInt32", types.UniTuple(types.int32, 2))
test_factory("TestTupleInt32Float32", types.Tuple([types.int32, types.float32]))

test_factory("Test1DArrayOfInt32", types.Array(types.int32, 1, 'C'),
             support_as_data=False)

test_factory("Test2DArrayOfInt32", types.Array(types.complex128, 2, 'C'),
             support_as_data=False)


class TestFunctionInfo(unittest.TestCase):
    def _test_as_arguments(self, fe_args):
        dmm = datamodel.defaultDataModelManager
        fe_ret = types.int32
        fi = datamodel.FunctionInfo(dmm, fe_ret, fe_args)

        module = ir.Module()
        fnty = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(module, fnty, name="test_arguments")
        builder = ir.IRBuilder()
        builder.position_at_end(function.append_basic_block())

        args = [ir.Constant(dmm.lookup(t).get_value_type(), None)
                for t in fe_args]

        values = fi.as_arguments(builder, args)
        asargs = fi.from_arguments(builder, values)

        self.assertEqual(len(asargs), len(fe_args))
        valtys = tuple([v.type for v in values])
        self.assertEqual(valtys, fi.argument_types)

        expect_types = [a.type for a in args]
        got_types = [a.type for a in asargs]

        self.assertEqual(expect_types, got_types)

        builder.ret_void()

        ll.parse_assembly(str(module))

    def test_int32_array_complex(self):
        fe_args = [types.int32,
                   types.Array(types.int32, 1, 'C'),
                   types.complex64]
        self._test_as_arguments(fe_args)

    def test_two_arrays(self):
        fe_args = [types.Array(types.int32, 1, 'C')] * 2
        self._test_as_arguments(fe_args)


if __name__ == '__main__':
    unittest.main()
