import unittest
from contextlib import contextmanager
from functools import cached_property

from numba import njit
from numba.core import errors, cpu, typing
from numba.core.descriptors import TargetDescriptor
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.retarget import BasicRetarget
from numba.core.extending import overload
from numba.core.target_extension import (
    dispatcher_registry,
    CPUDispatcher,
    CPU,
    target_registry,
    jit_registry,
)


# ------------ A custom target ------------

CUSTOM_TARGET = ".".join([__name__, "CustomCPU"])


class CustomCPU(CPU):
    """Extend from the CPU target
    """
    pass


# Implement a CustomCPU TargetDescriptor, this one borrows bits from the CPU
class CustomTargetDescr(TargetDescriptor):
    options = cpu.CPUTargetOptions

    @cached_property
    def _toplevel_target_context(self):
        # Lazily-initialized top-level target context, for all threads
        return cpu.CPUContext(self.typing_context, self._target_name)

    @cached_property
    def _toplevel_typing_context(self):
        # Lazily-initialized top-level typing context, for all threads
        return typing.Context()

    @property
    def target_context(self):
        """
        The target context for DPU targets.
        """
        return self._toplevel_target_context

    @property
    def typing_context(self):
        """
        The typing context for CPU targets.
        """
        return self._toplevel_typing_context


custom_target = CustomTargetDescr(CUSTOM_TARGET)


class CustomCPUDispatcher(CPUDispatcher):
    targetdescr = custom_target


target_registry[CUSTOM_TARGET] = CustomCPU
dispatcher_registry[target_registry[CUSTOM_TARGET]] = CustomCPUDispatcher


def custom_jit(*args, **kwargs):
    assert 'target' not in kwargs
    assert '_target' not in kwargs
    return njit(*args, _target=CUSTOM_TARGET, **kwargs)


jit_registry[target_registry[CUSTOM_TARGET]] = custom_jit

# ------------ For switching target ------------


class CustomCPURetarget(BasicRetarget):
    @property
    def output_target(self):
        return CUSTOM_TARGET

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target=CUSTOM_TARGET)(cpu_disp.py_func)
        return kernel


class TestRetargeting(unittest.TestCase):
    def setUp(self):
        # Generate fresh functions for each test method to avoid caching

        @njit(_target="cpu")
        def fixed_target(x):
            """
            This has a fixed target to "cpu".
            Cannot be used in CUSTOM_TARGET target.
            """
            return x + 10

        @njit
        def flex_call_fixed(x):
            """
            This has a flexible target, but uses a fixed target function.
            Cannot be used in CUSTOM_TARGET target.
            """
            return fixed_target(x) + 100

        @njit
        def flex_target(x):
            """
            This has a flexible target.
            Can be used in CUSTOM_TARGET target.
            """
            return x + 1000

        # Save these functions for use
        self.functions = locals()
        # Refresh the retarget function
        self.retarget = CustomCPURetarget()

    def switch_target(self):
        return TargetConfigurationStack.switch_target(self.retarget)

    @contextmanager
    def check_retarget_error(self):
        with self.assertRaises(errors.NumbaError) as raises:
            yield
        self.assertIn(f"{CUSTOM_TARGET} != cpu", str(raises.exception))

    def check_non_empty_cache(self):
        # Retargeting occurred. The cache must NOT be empty
        stats = self.retarget.cache.stats()
        # Because multiple function compilations are triggered, we don't know
        # precisely how many cache hit/miss there are.
        self.assertGreater(stats['hit'] + stats['miss'], 0)

    def test_case0(self):
        fixed_target = self.functions["fixed_target"]
        flex_target = self.functions["flex_target"]

        @njit
        def foo(x):
            x = fixed_target(x)
            x = flex_target(x)
            return x

        r = foo(123)
        self.assertEqual(r, 123 + 10 + 1000)
        # No retargeting occurred. The cache must be empty
        stats = self.retarget.cache.stats()
        self.assertEqual(stats, dict(hit=0, miss=0))

    def test_case1(self):
        flex_target = self.functions["flex_target"]

        @njit
        def foo(x):
            x = flex_target(x)
            return x

        with self.switch_target():
            r = foo(123)
        self.assertEqual(r, 123 + 1000)
        self.check_non_empty_cache()

    def test_case2(self):
        """
        The non-nested call into fixed_target should raise error.
        """
        fixed_target = self.functions["fixed_target"]
        flex_target = self.functions["flex_target"]

        @njit
        def foo(x):
            x = fixed_target(x)
            x = flex_target(x)
            return x

        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case3(self):
        """
        The nested call into fixed_target should raise error
        """
        flex_target = self.functions["flex_target"]
        flex_call_fixed = self.functions["flex_call_fixed"]

        @njit
        def foo(x):
            x = flex_call_fixed(x)  # calls fixed_target indirectly
            x = flex_target(x)
            return x

        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case4(self):
        """
        Same as case2 but flex_call_fixed() is invoked outside of CUSTOM_TARGET
        target before the switch_target.
        """
        flex_target = self.functions["flex_target"]
        flex_call_fixed = self.functions["flex_call_fixed"]

        r = flex_call_fixed(123)
        self.assertEqual(r, 123 + 100 + 10)

        @njit
        def foo(x):
            x = flex_call_fixed(x)  # calls fixed_target indirectly
            x = flex_target(x)
            return x

        with self.check_retarget_error():
            with self.switch_target():
                foo(123)

    def test_case5(self):
        """
        Tests overload resolution with target switching
        """

        def overloaded_func(x):
            pass

        @overload(overloaded_func, target=CUSTOM_TARGET)
        def ol_overloaded_func_custom_target(x):
            def impl(x):
                return 62830
            return impl

        @overload(overloaded_func, target='cpu')
        def ol_overloaded_func_cpu(x):
            def impl(x):
                return 31415
            return impl

        @njit
        def flex_resolve_overload(x):
            return

        @njit
        def foo(x):
            return x + overloaded_func(x)

        r = foo(123)
        self.assertEqual(r, 123 + 31415)

        with self.switch_target():
            r = foo(123)
            self.assertEqual(r, 123 + 62830)

        self.check_non_empty_cache()
