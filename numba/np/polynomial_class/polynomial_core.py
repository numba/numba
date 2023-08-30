from numba.extending import (models, register_model, as_numba_type,
                             type_callable, unbox, NativeValue,
                             make_attribute_wrapper, box, lower_builtin)
from numba.core import types, cgutils
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack
import numpy as np
from llvmlite import ir


@register_model(types.PolynomialType)
class PolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('coef', fe_type.coef),
            ('domain', fe_type.domain),
            ('window', fe_type.window)
            # Introduced in NumPy 1.24, maybe leave it out for now
            # ('symbol', types.string)
        ]
        super(PolynomialModel, self).__init__(dmm, fe_type, members)


# polynomial_type = types.PolynomialType()

as_numba_type.register(Polynomial, types.PolynomialType(np.array([0,1])))


@type_callable(Polynomial)
def type_polynomial(context):
    def typer(coef, domain=None, window=None):
        default_domain = types.Array(types.int64, 1, 'C')
        default_window = types.Array(types.int64, 1, 'C')
        default_coef = types.Array(types.double, 1, 'C')

        if isinstance(coef, types.Array) and \
                all([a is None for a in (domain, window)]):
            if coef.ndim == 1:
                return types.PolynomialType(default_coef,
                                            default_domain,
                                            default_window)
        elif all([isinstance(a, types.Array) for a in (coef, domain, window)]):
            if all([a.ndim == 1 for a in (coef, domain, window)]):
                return types.PolynomialType(default_coef, domain, window)
    return typer


make_attribute_wrapper(types.PolynomialType, 'coef', 'coef')
make_attribute_wrapper(types.PolynomialType, 'domain', 'domain')
make_attribute_wrapper(types.PolynomialType, 'window', 'window')
# Introduced in NumPy 1.24, maybe leave it out for now
# make_attribute_wrapper(types.PolynomialType, 'symbol', 'symbol')


@lower_builtin(Polynomial, types.Array)
def impl_polynomial1(context, builder, sig, args):
    import numpy as np

    def to_double(coef):
        return np.asarray(coef, dtype=np.double)

    def const_impl():
        return np.asarray([-1,1])

    typ = sig.return_type
    polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    sig_coef = sig.args[0].copy(dtype=types.double)(sig.args[0])
    coef_cast = context.compile_internal(builder, to_double, sig_coef, args)
    sig_domain = sig.args[0].copy(dtype=types.int64)()
    sig_window = sig.args[0].copy(dtype=types.int64)()
    domain_cast = context.compile_internal(builder, const_impl, sig_domain, ())
    window_cast = context.compile_internal(builder, const_impl, sig_window, ())
    polynomial.coef = coef_cast
    polynomial.domain = domain_cast
    polynomial.window = window_cast

    return polynomial._getvalue()


@lower_builtin(Polynomial, types.Array, types.Array, types.Array)
def impl_polynomial3(context, builder, sig, args):
    import numpy as np

    def to_double(coef):
        return np.asarray(coef, dtype=np.double)

    typ = sig.return_type
    polynomial = cgutils.create_struct_proxy(typ)(context, builder)

    coef_sig = sig.args[0].copy(dtype=types.double)(sig.args[0])
    coef_cast = context.compile_internal(builder,
                                         to_double, coef_sig,
                                         (args[0],))

    domain_cast = context.make_helper(builder, sig.args[1], value=args[1])
    window_cast = context.make_helper(builder, sig.args[2], value=args[2])

    i64 = ir.IntType(64)
    two = i64(2)

    s1 = builder.extract_value(domain_cast.shape, 0)
    s2 = builder.extract_value(window_cast.shape, 0)
    pred = builder.and_(
        builder.icmp_signed('!=', s1, two),
        builder.icmp_signed('!=', s2, two))

    with cgutils.if_unlikely(builder, pred):
        context.call_conv.return_user_exc(
            builder, ValueError,
            ("Domain and Window must be of length 2.",))

    polynomial.coef = coef_cast
    polynomial.domain = domain_cast._getvalue()
    polynomial.window = window_cast._getvalue()

    return polynomial._getvalue()


@unbox(types.PolynomialType)
def unbox_polynomial(typ, obj, c):
    """
    Convert a Polynomial object to a native polynomial structure.
    """
    is_error_ptr = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    with ExitStack() as stack:
        coef_obj = c.pyapi.object_getattr_string(obj, "coef")
        with cgutils.early_exit_if_null(c.builder, stack, coef_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        coef_native = c.unbox(typ.coef, coef_obj)
        c.pyapi.decref(coef_obj)
        with cgutils.early_exit_if(c.builder, stack, coef_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)

        domain_obj = c.pyapi.object_getattr_string(obj, "domain")
        with cgutils.early_exit_if_null(c.builder, stack, domain_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        domain_native = c.unbox(typ.domain, domain_obj)
        c.pyapi.decref(domain_obj)
        with cgutils.early_exit_if(c.builder, stack, domain_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)

        window_obj = c.pyapi.object_getattr_string(obj, "window")
        with cgutils.early_exit_if_null(c.builder, stack, window_obj):
            c.builder.store(cgutils.true_bit, is_error_ptr)
        window_native = c.unbox(typ.window, window_obj)
        c.pyapi.decref(window_obj)
        with cgutils.early_exit_if(c.builder, stack, window_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)

        polynomial.coef = coef_native.value
        polynomial.domain = domain_native.value
        polynomial.window = window_native.value

    return NativeValue(polynomial._getvalue(),
                       is_error=c.builder.load(is_error_ptr))


@box(types.PolynomialType)
def box_polynomial(typ, val, c):
    """
    Convert a native polynomial structure to a Polynomial object.
    """
    ret_ptr = cgutils.alloca_once(c.builder, c.pyapi.pyobj)
    fail_obj = c.pyapi.get_null_object()

    with ExitStack() as stack:
        polynomial = cgutils.create_struct_proxy(typ)(c.context, c.builder,
                                                      value=val)
        coef_obj = c.box(types.Array(types.double, 1, 'C'), polynomial.coef)
        with cgutils.early_exit_if_null(c.builder, stack, coef_obj):
            c.builder.store(fail_obj, ret_ptr)

        domain_obj = c.box(typ.domain, polynomial.domain)
        with cgutils.early_exit_if_null(c.builder, stack, domain_obj):
            c.builder.store(fail_obj, ret_ptr)

        window_obj = c.box(typ.window, polynomial.window)
        with cgutils.early_exit_if_null(c.builder, stack, window_obj):
            c.builder.store(fail_obj, ret_ptr)

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Polynomial))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(coef_obj)
            c.pyapi.decref(domain_obj)
            c.pyapi.decref(window_obj)
            c.builder.store(fail_obj, ret_ptr)

        res = c.pyapi.call_function_objargs(class_obj,
                                            (coef_obj, domain_obj, window_obj))
        c.pyapi.decref(coef_obj)
        c.pyapi.decref(domain_obj)
        c.pyapi.decref(window_obj)
        c.pyapi.decref(class_obj)
        c.builder.store(res, ret_ptr)

    return c.builder.load(ret_ptr)
