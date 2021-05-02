"""This tests the hardware extension API to ensure that rudimentary expected
behaviours are present and correct. It uses a piece of fake hardware, the
Dummy Processing Unit (DPU) to do this. The DPU borrows a lot from the CPU but
is part of the GPU class of hardware. The DPU target has deliberately strange
implementations of fundamental operations so as to make it identifiable in
testing."""

import unittest
from numba.tests.support import TestCase
import contextlib
import ctypes
import operator
import numpy as np
from numba import njit, types
from numba.extending import overload, intrinsic
from numba.core.extending_hardware import (
    JitDecorator,
    hardware_registry,
    dispatcher_registry,
    hardware_target,
    GPU,
    resolve_dispatcher_from_str,
)
from numba.core import utils, fastmathpass, errors
from numba.core.dispatcher import Dispatcher
from numba.core.descriptors import TargetDescriptor
from numba.core import cpu, typing, cgutils
from numba.core.base import BaseContext
from numba.core.compiler_lock import global_compiler_lock
from numba.core.utils import cached_property
from numba.core import callconv, decorators
from numba.core.codegen import CPUCodegen, JITCodeLibrary
from numba.core.callwrapper import PyCallWrapper
from numba.core.imputils import RegistryLoader, Registry
from numba import _dynfunc
import llvmlite.binding as ll
from llvmlite import ir as llir
from numba.core.runtime import rtsys

from numba.core import compiler
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreLowerStripPhis

# Define a new target, this hardware extends GPU, this places the DPU in the
# hardware hierarchy as a type of GPU.


class DPU(GPU):
    ...


# register the dpu hardware hierarchy token in the hardware registry, this
# permits lookup and reference in userspace by the string "dpu"
hardware_registry["dpu"] = DPU

# Create a JIT DPU codegen for the DPU target


class JITDPUCodegen(CPUCodegen):
    # This largely rips off the CPU for ease

    _library_class = JITCodeLibrary

    def _customize_tm_options(self, options):
        # Customize the target machine options.
        options["cpu"] = self._get_host_cpu_name()
        arch = ll.Target.from_default_triple().name
        if arch.startswith("x86"):
            reloc_model = "static"
        elif arch.startswith("ppc"):
            reloc_model = "pic"
        else:
            reloc_model = "default"
        options["reloc"] = reloc_model
        options["codemodel"] = "jitdefault"

        # Set feature attributes (such as ISA extensions)
        # This overrides default feature selection by CPU model above
        options["features"] = self._tm_features

        # Deal with optional argument to ll.Target.create_target_machine
        sig = utils.pysignature(ll.Target.create_target_machine)
        if "jit" in sig.parameters:
            # Mark that this is making a JIT engine
            options["jit"] = True

    def _customize_tm_features(self):
        # For JIT target, we will use LLVM to get the feature map
        return self._get_host_cpu_features()

    def _add_module(self, module):
        self._engine.add_module(module)

    def set_env(self, env_name, env):
        """Set the environment address.

        Update the GlobalVariable named *env_name* to the address of *env*.
        """
        gvaddr = self._engine.get_global_value_address(env_name)
        envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
        envptr[0] = ctypes.c_void_p(id(env))


# This is the function registry for the dpu, it just has one registry, this one!
dpu_function_registry = Registry()

# Implement a new context for the DPU target


class DPUContext(BaseContext):
    allow_dynamic_globals = True

    # Overrides
    def create_module(self, name):
        return self._internal_codegen._create_empty_module(name)

    @global_compiler_lock
    def init(self):
        self._internal_codegen = JITDPUCodegen("numba.exec")
        # Initialize NRT runtime
        rtsys.initialize(self)
        self.refresh()

    def refresh(self):
        registry = dpu_function_registry
        try:
            loader = self._registries[registry]
        except KeyError:
            loader = RegistryLoader(registry)
            self._registries[registry] = loader
        self.install_registry(registry)
        # Also refresh typing context, since @overload declarations can
        # affect it.
        self.typing_context.refresh()

    @property
    def target_data(self):
        return self._internal_codegen.target_data

    def codegen(self):
        return self._internal_codegen

    # Borrow the CPU call conv
    @cached_property
    def call_conv(self):
        return callconv.CPUCallConv(self)

    def get_env_body(self, builder, envptr):
        """
        From the given *envptr* (a pointer to a _dynfunc.Environment object),
        get a EnvBody allowing structured access to environment fields.
        """
        body_ptr = cgutils.pointer_add(
            builder, envptr, _dynfunc._impl_info["offsetof_env_body"]
        )
        return cpu.EnvBody(self, builder, ref=body_ptr, cast_ref=True)

    def get_env_manager(self, builder):
        envgv = self.declare_env_global(
            builder.module, self.get_env_name(self.fndesc)
        )
        envarg = builder.load(envgv)
        pyapi = self.get_python_api(builder)
        pyapi.emit_environment_sentry(
            envarg, debug_msg=self.fndesc.env_name,
        )
        env_body = self.get_env_body(builder, envarg)
        return pyapi.get_env_manager(self.environment, env_body, envarg)

    def get_generator_state(self, builder, genptr, return_type):
        """
        From the given *genptr* (a pointer to a _dynfunc.Generator object),
        get a pointer to its state area.
        """
        return cgutils.pointer_add(
            builder,
            genptr,
            _dynfunc._impl_info["offsetof_generator_state"],
            return_type=return_type,
        )

    def post_lowering(self, mod, library):
        if self.fastmath:
            fastmathpass.rewrite_module(mod, self.fastmath)

        library.add_linking_library(rtsys.library)

    def create_cpython_wrapper(
        self, library, fndesc, env, call_helper, release_gil=False
    ):
        wrapper_module = self.create_module("wrapper")
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = llir.Function(
            wrapper_module, fnty, fndesc.llvm_func_name
        )
        builder = PyCallWrapper(
            self,
            wrapper_module,
            wrapper_callee,
            fndesc,
            env,
            call_helper=call_helper,
            release_gil=release_gil,
        )
        builder.build()
        library.add_ir_module(wrapper_module)

    def create_cfunc_wrapper(self, library, fndesc, env, call_helper):
        # There's no cfunc wrapper on the dpu
        pass

    def get_executable(self, library, fndesc, env):
        """
        Returns
        -------
        (cfunc, fnptr)

        - cfunc
            callable function (Can be None)
        - fnptr
            callable function address
        - env
            an execution environment (from _dynfunc)
        """
        # Code generation
        fnptr = library.get_pointer_to_function(
            fndesc.llvm_cpython_wrapper_name
        )

        # Note: we avoid reusing the original docstring to avoid encoding
        # issues on Python 2, see issue #1908
        doc = "compiled wrapper for %r" % (fndesc.qualname,)
        cfunc = _dynfunc.make_function(
            fndesc.lookup_module(),
            fndesc.qualname.split(".")[-1],
            doc,
            fnptr,
            env,
            # objects to keepalive with the function
            (library,),
        )
        library.codegen.set_env(self.get_env_name(fndesc), env)
        return cfunc


# Nested contexts to help with isolatings bits of compilations
class _NestedContext(object):
    _typing_context = None
    _target_context = None

    @contextlib.contextmanager
    def nested(self, typing_context, target_context):
        old_nested = self._typing_context, self._target_context
        try:
            self._typing_context = typing_context
            self._target_context = target_context
            yield
        finally:
            self._typing_context, self._target_context = old_nested


# Implement a DPU TargetDescriptor, this one borrows bits from the CPU
class DPUTarget(TargetDescriptor):
    options = cpu.CPUTargetOptions
    _nested = _NestedContext()

    @utils.cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return DPUContext(self.typing_context, self._target_name)

    @utils.cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    def target_context(self):
        """
        The target context for DPU targets.
        """
        nested = self._nested._target_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        nested = self._nested._typing_context
        if nested is not None:
            return nested
        else:
            return self._toplevel_typing_context

    def nested_context(self, typing_context, target_context):
        """
        A context manager temporarily replacing the contexts with the
        given ones, for the current thread of execution.
        """
        return self._nested.nested(typing_context, target_context)


# Create a DPU target instance
dpu_target = DPUTarget("dpu")


# Declare a dispatcher for the DPU target
class DPUDispatcher(Dispatcher):
    targetdescr = dpu_target


# Register a dispatcher for the DPU target, a lot of the code uses this
# internally to work out what to do RE compilation
dispatcher_registry[hardware_registry["dpu"]] = DPUDispatcher

# Implement a dispatcher for the DPU target


class djit(JitDecorator):
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args):
        assert len(args) < 2
        if args:
            func = args[0]
        else:
            func = self._args[0]
        self.py_func = func
        # wrap in dispatcher
        return self.dispatcher_wrapper()

    def get_dispatcher(self):
        """
        Returns the dispatcher
        """
        return dispatcher_registry[hardware_registry["dpu"]]

    def dispatcher_wrapper(self):
        disp = self.get_dispatcher()
        # Parse self._kwargs here
        topt = {}
        if "nopython" in self._kwargs:
            topt["nopython"] = True

        # It would be easy to specialise the default compilation pipeline for
        # this target here.
        pipeline_class = compiler.Compiler
        if "pipeline_class" in self._kwargs:
            pipeline_class = self._kwargs["pipeline_class"]
        return disp(
            py_func=self.py_func,
            targetoptions=topt,
            pipeline_class=pipeline_class,
        )


# add it to the decorator registry, this is so e.g. @overload can look up a
# JIT function to do the compilation work.
decorators.jit_registry[hardware_registry["dpu"]] = djit

# The DPU target "knows" nothing, add in some primitives for basic things...

# need to register how to lower dummy for @intrinsic


@dpu_function_registry.lower_constant(types.Dummy)
def constant_dummy(context, builder, ty, pyval):
    return context.get_dummy_value()


# and how to deal with IntegerLiteral to Integer casts
@dpu_function_registry.lower_cast(types.IntegerLiteral, types.Integer)
def literal_int_to_number(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(
        builder, fromty.literal_type, fromty.literal_value,
    )
    return context.cast(builder, lit, fromty.literal_type, toty)


# and how to lower an Int constant
@dpu_function_registry.lower_constant(types.Integer)
def const_int(context, builder, ty, pyval):
    lty = context.get_value_type(ty)
    return lty(pyval)


# and tell the DPU how to lower a float constant
@dpu_function_registry.lower_constant(types.Float)
def const_float(context, builder, ty, pyval):
    lty = context.get_value_type(ty)
    return lty(pyval)


# The DPU actually subtracts when it's asked to 'add'!
@intrinsic(hardware="dpu")
def intrin_add(tyctx, x, y):
    sig = x(x, y)

    def codegen(cgctx, builder, tyargs, llargs):
        return builder.sub(*llargs)

    return sig, codegen


# Use extending.overload API to register 'add', call the dpu specific intrinsic
@overload(operator.add, hardware="dpu")
def ol_add(x, y):
    if isinstance(x, types.Integer) and isinstance(y, types.Integer):

        def impl(x, y):
            return intrin_add(x, y)

        return impl


class TestHardwareHierarchySelection(TestCase):
    """This tests that the hardware hierarchy is scanned in the right order,
    that appropriate functions are selected based on what's available and that
    the DPU target is distinctly different to the CPU"""

    def test_generic(self):
        def my_func(x):
            pass

        # Can be used by both CPU and DPU
        @overload(my_func, hardware="generic")
        def ol_my_func1(x):
            def impl(x):
                return 1 + x

            return impl

        @djit()
        def dpu_foo():
            return my_func(7)

        @njit()
        def cpu_foo():
            return my_func(7)

        self.assertPreciseEqual(dpu_foo(), -6)  # DPU subtracts
        self.assertPreciseEqual(cpu_foo(), 8)  # CPU adds

    def test_specialise_gpu(self):
        def my_func(x):
            pass

        # Can be used by both CPU and DPU
        @overload(my_func, hardware="generic")
        def ol_my_func1(x):
            def impl(x):
                return 1 + x

            return impl

        # Should be used by the DPU if there's no dpu specific one
        @overload(my_func, hardware="gpu")
        def ol_my_func2(x):
            def impl(x):
                return 10 + x

            return impl

        @djit()
        def dpu_foo():
            return my_func(7)

        @njit()
        def cpu_foo():
            return my_func(7)

        # DPU chooses the ol_my_func2 as it's most specific, and DPU subtracts
        # for addition, so 10 + x -> 10 - 7 -> 3
        self.assertPreciseEqual(dpu_foo(), 3)
        # CPU uses the generic one function ol_my_func1 and adds
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_specialise_dpu(self):
        def my_func(x):
            pass

        # Can be used by both CPU and DPU
        @overload(my_func, hardware="generic")
        def ol_my_func1(x):
            def impl(x):
                return 1 + x

            return impl

        # Should be used by the DPU if there's no dpu specific one
        @overload(my_func, hardware="gpu")
        def ol_my_func2(x):
            def impl(x):
                return 10 + x

            return impl

        # Should be used by the DPU only
        @overload(my_func, hardware="dpu")
        def ol_my_func3(x):
            def impl(x):
                return 100 + x

            return impl

        @djit()
        def dpu_foo():
            return my_func(7)

        @njit()
        def cpu_foo():
            return my_func(7)

        # DPU chooses the ol_my_func3 as it's most specific, and DPU subtracts
        # for addition, so 100 + x -> 100 - 7 -> 93
        self.assertPreciseEqual(dpu_foo(), 93)
        # CPU uses the generic one function ol_my_func1 and adds
        self.assertPreciseEqual(cpu_foo(), 8)

    def test_no_specialisation_found(self):

        def my_func(x):
            pass

        # only create a cuda specialisation
        @overload(my_func, hardware='cuda')
        def ol_my_func_cuda(x):
            return lambda x: None

        @djit(nopython=True)
        def dpu_foo():
            my_func(1)

        with self.assertRaises(errors.TypingError) as raises:
            dpu_foo()

        msgs = ["Function resolution cannot find any matches for function",
                "test_no_specialisation_found.<locals>.my_func",
                "for the current hardware:",
                "'numba.tests.test_hardware_extension.DPU'"]

        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

    def test_dpu_registry(self):
        """Checks that the DPU registry only contains the things added"""
        self.assertFalse(dpu_function_registry.functions)
        self.assertFalse(dpu_function_registry.getattrs)
        # int literal -> int cast is registered
        self.assertEqual(len(dpu_function_registry.casts), 1)
        # int, float and dummy constants are registered
        self.assertEqual(len(dpu_function_registry.constants), 3)

    def test_invalid_target_jit(self):

        with self.assertRaises(ValueError) as raises:
            @njit(_target='invalid_silicon')
            def foo():
                pass

            foo()

        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_invalid_target_overload(self):

        def bar():
            pass

        # This is a typing error at present as it fails during typing when the
        # overloads are walked.
        with self.assertRaises(errors.TypingError) as raises:
            @overload(bar, hardware='invalid_silicon')
            def ol_bar():
                return lambda : None

            @njit
            def foo():
                bar()

            foo()

        msg = "No target is registered against 'invalid_silicon'"
        self.assertIn(msg, str(raises.exception))

    def test_intrinsic_selection(self):
        """
        Test to make sure that targets can share generic implementations and
        cannot reach implementations that are not in their hardware hierarchy.
        """

        # NOTE: The actual operation performed by these functions is irrelevant
        @intrinsic(hardware="generic")
        def intrin_math_generic(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.mul(*llargs)

            return sig, codegen

        @intrinsic(hardware="dpu")
        def intrin_math_dpu(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.sub(*llargs)

            return sig, codegen

        @intrinsic(hardware="cpu")
        def intrin_math_cpu(tyctx, x, y):
            sig = x(x, y)

            def codegen(cgctx, builder, tyargs, llargs):
                return builder.add(*llargs)

            return sig, codegen

        # CPU can use the CPU version
        @njit
        def cpu_foo_specific():
            return intrin_math_cpu(3, 4)

        self.assertEqual(cpu_foo_specific(), 7)

        # CPU can use the 'generic' version
        @njit
        def cpu_foo_generic():
            return intrin_math_generic(3, 4)

        self.assertEqual(cpu_foo_generic(), 12)

        # CPU cannot use the 'dpu' version
        @njit
        def cpu_foo_dpu():
            return intrin_math_dpu(3, 4)

        with self.assertRaises(errors.TypingError) as raises:
            cpu_foo_dpu()

        msgs = ["Function resolution cannot find any matches for function",
                "intrinsic intrin_math_dpu",
                "for the current hardware",]
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))

        # DPU can use the DPU version
        @djit(nopython=True)
        def dpu_foo_specific():
            return intrin_math_dpu(3, 4)

        self.assertEqual(dpu_foo_specific(), -1)

        # DPU can use the 'generic' version
        @djit(nopython=True)
        def dpu_foo_generic():
            return intrin_math_generic(3, 4)

        self.assertEqual(dpu_foo_generic(), 12)

        # DPU cannot use the 'cpu' version
        @djit(nopython=True)
        def dpu_foo_cpu():
            return intrin_math_cpu(3, 4)

        with self.assertRaises(errors.TypingError) as raises:
            dpu_foo_cpu()

        msgs = ["Function resolution cannot find any matches for function",
                "intrinsic intrin_math_cpu",
                "for the current hardware",]
        for msg in msgs:
            self.assertIn(msg, str(raises.exception))


class TestHardwareOffload(TestCase):
    """In this use case the CPU compilation pipeline is extended with a new
     compilation pass that runs just prior to lowering. The pass looks for
     function calls and when it finds one it sees if there's a DPU function
     available that is a valid overload for the function call. If there is one
     then it swaps the CPU implementation out for a DPU implementation. This
     producing an "offload" effect.
    """

    def test_basic_offload(self):

        _DEBUG = False

        # This is the DPU function for sin, it'll return a pi-like constant
        @overload(np.sin, hardware="dpu")
        def ol_np_sin_DPU(x):
            def dpu_sin_impl(x):
                return 314159.0

            return dpu_sin_impl

        # Check the DPU reports the correct overload value
        @djit(nopython=True)
        def foo(x):
            return np.sin(x)

        self.assertPreciseEqual(foo(5), 314159.0)

        # Check the CPU call is correct

        @njit
        def foo(x):
            return np.sin(x)

        self.assertPreciseEqual(foo(5), np.sin(5))

        @register_pass(mutates_CFG=False, analysis_only=False)
        class DispatcherSwitcher(FunctionPass):
            _name = "DispatcherSwitcher"

            def __init__(self):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                func_ir = state.func_ir
                mutated = False
                for blk in func_ir.blocks.values():
                    # find the assignment nodes in the block and walk them, if
                    # there's a DPU version then swap out for a call to that
                    for call in blk.find_exprs("call"):
                        function = state.typemap[call.func.name]
                        tname = "dpu"

                        # Note: `hardware_target` context driven compilation can
                        # be done here, the DPU target is in use.
                        with hardware_target(tname):
                            try:
                                sig = function.get_call_type(
                                    state.typingctx,
                                    state.calltypes[call].args,
                                    {},
                                )
                                disp = resolve_dispatcher_from_str(tname)
                                # force compile check
                                hw_ctx = disp.targetdescr.target_context
                                hw_ctx.get_function(function, sig)
                            except Exception as e:
                                if _DEBUG:
                                    msg = (
                                        f"Failed to find and compile an "
                                        f"overload for {function} for {tname} "
                                        f"due to {e}"
                                    )
                                    print(msg)
                                continue

                            # This is a necessary hack at present so as to
                            # generate code into the same library. I.e. the DPU
                            # target is going to do code gen into the CPUs lib.
                            hw_ctx._codelib_stack = (
                                state.targetctx._codelib_stack
                            )

                            # All is good, so switch IR node for one targeting
                            # this hardware. Should generate this, but for now
                            # just mutate as:
                            # ir.Expr.call(call.func, call.args, call.kws,
                            #              call.loc, hardware='dpu')
                            call.hardware = tname
                            mutated = True
                # return True if the IR was mutated, False if not.
                return mutated

        # DPU compiler pipeline, compiles with offload to the DPU target
        class DPUOffloadCompiler(CompilerBase):
            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(DispatcherSwitcher, PreLowerStripPhis)
                pm.finalize()
                return [pm]

        # Now compile for CPU, but with the DispatcherSwitcher pass in place
        # that switches CPU calls for DPU calls
        @njit(pipeline_class=DPUOffloadCompiler)
        def foo(x):
            return np.sin(x), np.cos(x)  # np.sin is DPU, np.cos is CPU

        self.assertPreciseEqual(foo(5), (314159.0, np.cos(5)))


if __name__ == "__main__":
    unittest.main()
