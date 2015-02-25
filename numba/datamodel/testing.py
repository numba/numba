from __future__ import print_function, absolute_import

from llvmlite import ir
from llvmlite import binding as ll

from numba import datamodel
from numba import unittest_support as unittest


class DataModelTester(unittest.TestCase):
    """
    Test the implementation of a DataModel for a frontend type.
    """
    fe_type = NotImplemented

    def setUp(self):
        self.module = ir.Module()
        self.datamodel = datamodel.defaultDataModelManager[self.fe_type]

    def test_as_arg(self):
        """
        - Is as_arg() and from_arg() implemented?
        - Are they the inverse of each other?
        """
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
        """
        - Is as_return() and from_return() implemented?
        - Are they the inverse of each other?
        """
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


class SupportAsDataMixin(object):
    """Test as_data() and from_data()
    """

    def test_as_data(self):
        fnty = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(self.module, fnty, name="test_as_data")
        builder = ir.IRBuilder()
        builder.position_at_end(function.append_basic_block())

        undef_value = ir.Constant(self.datamodel.get_value_type(), None)
        data = self.datamodel.as_data(builder, undef_value)
        self.assertIsNot(data, NotImplemented,
                         "as_data returned NotImplemented")

        self.assertEqual(data.type, self.datamodel.get_data_type())

        rev_value = self.datamodel.from_data(builder, data)
        self.assertEqual(rev_value.type,
                         self.datamodel.get_value_type())

        builder.ret_void()  # end function

        # Ensure valid LLVM generation
        materialized = ll.parse_assembly(str(self.module))
        print(materialized)


class NotSupportAsDataMixin(object):
    """Ensure as_data() and from_data() returns NotImplemented
    """

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


class DataModelTester_SupportAsDataMixin(DataModelTester,
                                         SupportAsDataMixin):
    pass


class DataModelTester_NotSupportAsDataMixin(DataModelTester,
                                            NotSupportAsDataMixin):
    pass


def test_factory(support_as_data=True):
    """A helper for returning a unittest TestCase for testing
    """
    if support_as_data:
        return DataModelTester_SupportAsDataMixin
    else:
        return DataModelTester_NotSupportAsDataMixin
