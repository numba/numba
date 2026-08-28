import numba as nb
from numba import types
from numba.core import cgutils
from numba.extending import (typeof_impl, register_model,
                             models, infer_getattr,
                             lower_builtin)
from numba.core.imputils import lower_constant
from numba.core.typing.templates import bound_function, AttributeTemplate
import inspect
import unittest


class TestBoundFunction(unittest.TestCase):
    def test_bound_function(self):
        class MyTypeObj():
            pass

        class MyType(types.Type):
            def __init__(self, name):
                super().__init__(name)

        @register_model(MyType)
        class MyTypeModel(models.StructModel):
            def __init__(self, dmm, fe_type):
                members = [
                    ('id', types.int32),
                ]
                super(MyTypeModel, self).__init__(dmm, fe_type, members)

        @typeof_impl.register(MyTypeObj)
        def typeof_index(val, c):
            return MyType("mytype")

        @lower_constant(MyType)
        def lower_mytype_constant(context, builder, typ, pyval):
            proxy = cgutils.create_struct_proxy(typ)(context, builder)
            proxy.id = context.get_constant(types.int32, 0)
            return proxy._getvalue()

        def process_args(self_ty, args, kwargs):
            is_lit = [isinstance(a, types.Literal) for a in args]
            is_kws = [isinstance(kwargs[key], types.Literal) for key in kwargs]
            all_are_literal = all(is_lit + is_kws)
            self.assertEqual(all_are_literal, True)

            def_args = [(f"x{x}", args[x]) for x in range(len(args))]
            def_kws = [(key, kwargs[key]) for key in kwargs]
            def_self_arg = []
            def_self_arg = [("self", self_ty)]

            ptype = inspect.Parameter.POSITIONAL_OR_KEYWORD
            params = [inspect.Parameter(def_self_arg[0][0],ptype)]
            for parg in def_args:
                params.append(inspect.Parameter(parg[0],ptype))

            for parg in def_kws:
                params.append(inspect.Parameter(parg[0],ptype))

            pysig = inspect.Signature(params)
            sig_args = [x[1] for x in def_self_arg + def_args + def_kws]

            sig = nb.types.int32(*sig_args)
            sig = sig.replace(pysig=pysig)

            return sig

        @infer_getattr
        class MyObjAttribute(AttributeTemplate):
            key = MyType

            @bound_function("myobj.test1", prefer_literal=True)
            def resolve_test1(self, ty, args, kws):
                sig = process_args(ty, args, kws)
                if (sig is None):
                    return

                @lower_builtin("myobj.test1", *sig.args)
                def lower_test1(context, builder, sig, args):
                    _sum = context.get_constant(sig.return_type, 0)
                    for arg, typ in zip(args[1:], sig.args[1:]):
                        val = context.cast(builder, arg, typ, sig.return_type)
                        _sum = builder.add(_sum, val)
                    return _sum
                return sig.as_method()

        myobj = MyTypeObj()

        @nb.njit(nogil=True)
        def main():
            return myobj.test1(5, 6, 2, 3, k=33, j=9, i=25)
        result_sum = main()
        self.assertEqual(result_sum, 83)
