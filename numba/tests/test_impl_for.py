import unittest
from functools import partial

from numba.tests.support import TestCase
from numba.core.extending import overload, impl_for
from numba.core import types
from numba.core.datamodel import models
from numba.core.registry import cpu_target
from numba.core.errors import CompilerError


class TestImplForTypeResolution(TestCase):
    def test_override_by_array_subtype(self):
        def foo():
            # function to be overloaded
            pass

        class AltArray(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"AltArray({self.name})"

        class Tensor(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"Tensor({self.name})"

        class DeepTensor(Tensor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"DeepTensor({self.name})"

        models.register_default(AltArray)(models.ArrayModel)
        models.register_default(Tensor)(models.ArrayModel)
        models.register_default(DeepTensor)(models.ArrayModel)

        @overload(foo, use_impl_for=True)
        def ov_foo_base(x, y):
            if isinstance(x, types.Array) and isinstance(y, types.Array):

                @impl_for(types.Array)
                def impl(x, y):
                    return "base"

                return impl

        @overload(foo, use_impl_for=True)
        def ov_foo_tensor(x, y):
            if isinstance(x, types.Array) and isinstance(y, types.Array):
                if isinstance(x, Tensor) or isinstance(y, Tensor):

                    @impl_for(Tensor)
                    def impl(x, y):
                        return "tensor"

                    return impl

        tyctx = cpu_target.typing_context
        tyctx.refresh()

        array_t = types.Array(types.intp, 2, "A")
        tensor_t = Tensor(types.intp, 2, "A")
        altarray_t = AltArray(types.intp, 2, "A")
        deeptensor_t = DeepTensor(types.intp, 2, "A")
        fnty = tyctx.resolve_value_type(foo)

        check = partial(self.check, tyctx, fnty)
        # Base
        check((array_t, array_t), types.literal("base"))
        # Tensor
        check((tensor_t, tensor_t), types.literal("tensor"))
        # AltArray
        check((altarray_t, altarray_t), types.literal("base"))
        # DeepTensor
        check((deeptensor_t, deeptensor_t), types.literal("tensor"))
        # Mix Tensor
        check((array_t, tensor_t), types.literal("tensor"))
        check((deeptensor_t, tensor_t), types.literal("tensor"))
        check((deeptensor_t, altarray_t), types.literal("tensor"))

    def test_ambiguous_open(self):
        def foo():
            # function to be overloaded
            pass

        class Tensor(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"Tensor({self.name})"

        models.register_default(Tensor)(models.ArrayModel)

        @overload(foo)
        def ov_foo_base(x, y):
            if isinstance(x, types.Array) and isinstance(y, types.Array):

                def impl(x, y):
                    return "base"

                return impl

        @overload(foo, use_impl_for=True)
        def ov_foo_tensor(x, y):
            if isinstance(x, Tensor) and isinstance(y, Tensor):
                # intentionally left open (no impl_for)
                def impl(x, y):
                    return "tensor"

                return impl

        tyctx = cpu_target.typing_context
        tyctx.refresh()

        tensor_t = Tensor(types.intp, 2, "A")

        fnty = tyctx.resolve_value_type(foo)

        # When one of the matching overload opt-in to use_impl_for
        self.check_fail(
            tyctx,
            fnty,
            (tensor_t, tensor_t),
            Exception,
            "ambiguous open versions",
        )

    def test_ambiguous_impl_for(self):
        def foo():
            # function to be overloaded
            pass

        class Tensor(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"Tensor({self.name})"

        models.register_default(Tensor)(models.ArrayModel)

        @overload(foo, use_impl_for=True)
        def ov_foo_base(x, y):
            if isinstance(x, types.Array) and isinstance(y, types.Array):

                @impl_for(types.Array)
                def impl(x, y):
                    return "base"

                return impl

        @overload(foo, use_impl_for=True)
        def ov_foo_tensor(x, y):
            if isinstance(x, Tensor) and isinstance(y, Tensor):

                @impl_for(types.Array)
                def impl(x, y):
                    return "tensor"

                return impl

        tyctx = cpu_target.typing_context
        tyctx.refresh()

        tensor_t = Tensor(types.intp, 2, "A")

        fnty = tyctx.resolve_value_type(foo)

        # When one of the matching overload opt-in to use_impl_for
        self.check_fail(
            tyctx,
            fnty,
            (tensor_t, tensor_t),
            Exception,
            "ambiguous with specialized versions",
        )

    def test_disjoint_impl_for_types(self):
        def foo():
            # function to be overloaded
            pass

        class Tensor(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"Tensor({self.name})"

        models.register_default(Tensor)(models.ArrayModel)

        @overload(foo, use_impl_for=True)
        def ov_foo_base(x, y):
            if isinstance(x, types.Array) and isinstance(y, types.Array):

                @impl_for(types.Array)
                def impl(x, y):
                    return "base"

                return impl

        @overload(foo, use_impl_for=True)
        def ov_foo_tensor(x, y):
            # intentionally accept if either x or y is a Tensor
            if isinstance(x, Tensor) or isinstance(y, Tensor):

                @impl_for(Tensor)
                def impl(x, y):
                    return "tensor"

                return impl

        @overload(foo, use_impl_for=True)
        def ov_foo_disjoint(x, y):
            if isinstance(x, Tensor) and isinstance(y, types.Integer):

                @impl_for(types.Integer)
                def impl(x, y):
                    return "int"

                return impl

        tyctx = cpu_target.typing_context
        tyctx.refresh()

        array_t = types.Array(types.intp, 2, "A")
        tensor_t = Tensor(types.intp, 2, "A")
        int_t = types.intp
        fnty = tyctx.resolve_value_type(foo)

        check = partial(self.check, tyctx, fnty)

        check((array_t, array_t), types.literal("base"))
        check((tensor_t, tensor_t), types.literal("tensor"))
        self.check_fail(
            tyctx,
            fnty,
            (tensor_t, int_t),
            CompilerError,
            "Not all signatures have a common `impl_for` ancestry",
        )

    def check_dense_sparse(self, fix_by_more_specific):
        class Dense(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"dense.{self.name}"

        class Sparse(types.Array):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.name = f"sparse.{self.name}"

        def matmul(a, b):
            pass

        @overload(matmul)
        def ov_matmul(a, b):
            if isinstance(a, types.Array) and isinstance(b, types.Array):

                def impl(a, b):
                    return "base"

                return impl

        @overload(matmul, use_impl_for=True)
        def ov_matmul_dense(a, b):
            if isinstance(a, types.Array) and isinstance(b, types.Array):
                if isinstance(a, Dense) or isinstance(b, Dense):

                    @impl_for(Dense)
                    def impl(a, b):
                        return "dense@dense"

                    return impl

        @overload(matmul, use_impl_for=True)
        def ov_matmul_sparse(a, b):
            if isinstance(a, types.Array) and isinstance(b, types.Array):
                if isinstance(a, Sparse) or isinstance(b, Sparse):

                    @impl_for(Sparse)
                    def impl(a, b):
                        return "sparse@sparse"

                    return impl

        if fix_by_more_specific:
            class DenseSparse(Dense, Sparse):
                pass

            @overload(matmul, use_impl_for=True)
            def ov_matmul_dense_sparse(a, b):
                if isinstance(a, Dense) and isinstance(b, Sparse):

                    @impl_for(DenseSparse)
                    def impl(a, b):
                        return "dense@sparse"

                    return impl

            @overload(matmul, use_impl_for=True)
            def ov_matmul_sparse_dense(a, b):
                if isinstance(a, Sparse) and isinstance(b, Dense):

                    @impl_for(DenseSparse)
                    def impl(a, b):
                        return "sparse@dense"

                    return impl

        models.register_default(Dense)(models.ArrayModel)
        models.register_default(Sparse)(models.ArrayModel)

        tyctx = cpu_target.typing_context
        tyctx.refresh()

        array_t = types.Array(types.intp, 2, "A")
        dense_t = Dense(types.intp, 2, "A")
        sparse_t = Sparse(types.intp, 2, "A")

        fnty = tyctx.resolve_value_type(matmul)

        def check(args, expect):
            self.check(tyctx, fnty, args, types.literal(expect))

        # basic
        check((array_t, array_t), "base")
        check((dense_t, array_t), "dense@dense")
        check((dense_t, dense_t), "dense@dense")
        check((sparse_t, array_t), "sparse@sparse")
        check((sparse_t, sparse_t), "sparse@sparse")

        def dillenma():
            check((dense_t, sparse_t), "dense@sparse")
            check((sparse_t, dense_t), "sparse@dense")

        if fix_by_more_specific:
            dillenma()
        else:
            with self.assertRaises(CompilerError) as raises:
                dillenma()
            self.assertIn(
                "Not all signatures have a common `impl_for` ancestry",
                str(raises.exception),
            )

    def test_dense_sparse_ok(self):
        self.check_dense_sparse(fix_by_more_specific=True)

    def test_dense_sparse_expected_failure(self):
        self.check_dense_sparse(fix_by_more_specific=False)

    def check(self, tyctx, fnty, argtys, expect):
        out_sig = tyctx.resolve_function_type(fnty, argtys, {})
        self.assertEqual(out_sig.return_type, expect)

    def check_fail(self, tyctx, fnty, argtys, errcls, errmsg):
        with self.assertRaises(errcls) as raises:
            tyctx.resolve_function_type(fnty, argtys, {})
        self.assertIn(errmsg, str(raises.exception))


if __name__ == "__main__":
    unittest.main()
