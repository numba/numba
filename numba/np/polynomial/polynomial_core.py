from numba.extending import (models, register_model, as_numba_type,
                             type_callable, unbox, NativeValue,
                             make_attribute_wrapper, box, lower_builtin)
from numba.core import types, cgutils
from numba.core.typing import infer_global
from numba.core.typing.templates import AbstractTemplate
from numpy.polynomial.polynomial import Polynomial
from contextlib import ExitStack


@register_model(types.PolynomialType)
class PolynomialModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('coef', types.Array(types.double, 1, 'C')),
            # ('domain', types.Array(types.double, 1, 'C')),
            # ('window', types.Array(types.double, 1, 'C'))
            # Introduced in NumPy 1.24, maybe leave it out for now
            # ('symbol', types.string)
        ]
        super(PolynomialModel, self).__init__(dmm, fe_type, members)


polynomial_type = types.PolynomialType()

as_numba_type.register(Polynomial, polynomial_type)


# @type_callable(Polynomial)
# def type_polynomial(context):
#     def typer(coef):
#         if isinstance(coef, types.Array):
#             return polynomial_type
#     return typer

@infer_global(types.PolynomialType)
class Poly(AbstractTemplate):

    def generic(self, args, kws):
        breakpoint()
        print(args)
        print(kws)

make_attribute_wrapper(types.PolynomialType, 'coef', 'coef')
# make_attribute_wrapper(types.PolynomialType, 'domain', 'domain')
# make_attribute_wrapper(types.PolynomialType, 'window', 'window')
# Introduced in NumPy 1.24, maybe leave it out for now
# make_attribute_wrapper(types.PolynomialType, 'symbol', 'symbol')


@lower_builtin(Polynomial, types.Array)
def impl_polynomial1(context, builder, sig, args):
    import numpy as np

    def impl(coef):
        return np.asarray(coef, dtype=np.double)

    typ = sig.return_type
    polynomial = cgutils.create_struct_proxy(typ)(context, builder)
    sig2 = sig.args[0].copy(dtype=types.double)(sig.args[0])
    coef_cast = context.compile_internal(builder, impl, sig2, args)
    polynomial.coef = coef_cast

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
        coef_native = c.unbox(types.Array(types.double, 1, 'C'), coef_obj)
        c.pyapi.decref(coef_obj)

        with cgutils.early_exit_if(c.builder, stack, coef_native.is_error):
            c.builder.store(cgutils.true_bit, is_error_ptr)

        polynomial.coef = coef_native.value

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

        class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Polynomial))
        with cgutils.early_exit_if_null(c.builder, stack, class_obj):
            c.pyapi.decref(coef_obj)
            c.builder.store(fail_obj, ret_ptr)

        # NOTE: The result of this call is not checked as the clean up
        # has to occur regardless of whether it is successful. If it
        # fails `res` is set to NULL and a Python exception is set.
        res = c.pyapi.call_function_objargs(class_obj, (coef_obj, ))
        c.pyapi.decref(coef_obj)
        c.pyapi.decref(class_obj)
        c.builder.store(res, ret_ptr)

    return c.builder.load(ret_ptr)
