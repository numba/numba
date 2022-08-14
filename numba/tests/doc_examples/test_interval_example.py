"""
This test is used by `docs/source/extending/interval-example.rst`.

The "magictoken" comments are used as markers for the beginning and ending of
example code.
"""
import unittest


class IntervalExampleTest(unittest.TestCase):

    def test_interval_class_usage(self):
        # magictoken.interval_py_class.begin
        class Interval:
            """
            A half-open interval on the real number line.
            """

            def __init__(self, lo, hi):
                self.lo = lo
                self.hi = hi

            def __repr__(self):
                return f'Interval({self.lo:f}, {self.hi:f})'

            @property
            def width(self):
                return self.hi - self.lo
        # magictoken.interval_py_class.end

        # magictoken.interval_type_class.begin
        from numba import types

        class IntervalType(types.Type):
            def __init__(self):
                super().__init__(name='Interval')

        interval_type = IntervalType()
        # magictoken.interval_type_class.end

        # magictoken.interval_typeof_register.begin
        from numba.extending import typeof_impl

        @typeof_impl.register(Interval)
        def typeof_index(val, c):
            return interval_type
        # magictoken.interval_typeof_register.end

        # magictoken.numba_type_register.begin
        from numba.extending import as_numba_type

        as_numba_type.register(Interval, interval_type)
        # magictoken.numba_type_register.end

        # magictoken.numba_type_callable.begin
        from numba.extending import type_callable

        @type_callable(Interval)
        def type_interval(context):
            def typer(lo, hi):
                if isinstance(lo, types.Float) and isinstance(hi, types.Float):
                    return interval_type
            return typer
        # magictoken.numba_type_callable.end

        # magictoken.interval_model.begin
        from numba.extending import models, register_model

        @register_model(IntervalType)
        class IntervalModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                members = [('lo', types.float64),
                           ('hi', types.float64),]
                models.StructModel.__init__(self, dmm, fe_type, members)
        # magictoken.interval_model.end

        # magictoken.interval_attribute_wrapper.begin
        from numba.extending import make_attribute_wrapper

        make_attribute_wrapper(IntervalType, 'lo', 'lo')
        make_attribute_wrapper(IntervalType, 'hi', 'hi')
        # magictoken.interval_attribute_wrapper.end

        # magictoken.interval_overload_attribute.begin
        from numba.extending import overload_attribute

        @overload_attribute(IntervalType, "width")
        def get_width(interval):
            def getter(interval):
                return interval.hi - interval.lo
            return getter
        # magictoken.interval_overload_attribute.end

        # magictoken.interval_lower_builtin.begin
        from numba.extending import lower_builtin
        from numba.core import cgutils

        @lower_builtin(Interval, types.Float, types.Float)
        def impl_interval(context, builder, sig, args):
            typ = sig.return_type
            lo, hi = args
            interval = cgutils.create_struct_proxy(typ)(context, builder)
            interval.lo = lo
            interval.hi = hi
            return interval._getvalue()
        # magictoken.interval_lower_builtin.end

        # magictoken.interval_unbox.begin
        from numba.extending import unbox, NativeValue

        @unbox(IntervalType)
        def unbox_interval(typ, obj, c):
            """
            Convert a Interval object to a native interval structure.
            """
            lo_obj = c.pyapi.object_getattr_string(obj, "lo")
            hi_obj = c.pyapi.object_getattr_string(obj, "hi")
            interval = cgutils.create_struct_proxy(typ)(c.context, c.builder)
            interval.lo = c.pyapi.float_as_double(lo_obj)
            interval.hi = c.pyapi.float_as_double(hi_obj)
            c.pyapi.decref(lo_obj)
            c.pyapi.decref(hi_obj)
            is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
            return NativeValue(interval._getvalue(), is_error=is_error)
        # magictoken.interval_unbox.end

        # magictoken.interval_box.begin
        from numba.extending import box

        @box(IntervalType)
        def box_interval(typ, val, c):
            """
            Convert a native interval structure to an Interval object.
            """
            interval = cgutils.create_struct_proxy(typ)(c.context,
                                                        c.builder,
                                                        value=val)
            lo_obj = c.pyapi.float_from_double(interval.lo)
            hi_obj = c.pyapi.float_from_double(interval.hi)
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Interval))
            res = c.pyapi.call_function_objargs(class_obj, (lo_obj, hi_obj))
            c.pyapi.decref(lo_obj)
            c.pyapi.decref(hi_obj)
            c.pyapi.decref(class_obj)
            return res
        # magictoken.interval_box.end

        # magictoken.interval_usage.begin
        from numba import njit

        @njit
        def inside_interval(interval, x):
            return interval.lo <= x < interval.hi

        @njit
        def interval_width(interval):
            return interval.width

        @njit
        def sum_intervals(i, j):
            return Interval(i.lo + j.lo, i.hi + j.hi)
        # magictoken.interval_usage.end

        def check_equal_intervals(x, y):
            self.assertIsInstance(x, Interval)
            self.assertIsInstance(y, Interval)
            self.assertEqual(x.lo, y.lo)
            self.assertEqual(x.hi, y.hi)

        a = Interval(2, 3)
        b = Interval(4, 5)
        c = Interval(6, 8)

        # Test box-unbox
        return_func = njit(lambda x: x)
        check_equal_intervals(a, return_func(a))

        # Test .width attribute
        self.assertEqual(a.width, interval_width(a))

        # Test .low and .high usage
        self.assertFalse(inside_interval(a, 5))

        # Test native Interval constructor
        check_equal_intervals(c, sum_intervals(a, b))


if __name__ == '__main__':
    unittest.main()
