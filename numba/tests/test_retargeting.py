import unittest

from contextlib import contextmanager

from numba import njit
from numba.core import errors
from numba.core.dispatcher import TargetConfig
from numba.core.retarget import BasicRetarget
from numba.core.target_extension import (
    dispatcher_registry,
    CPUDispatcher,
    CPU,
    target_registry,
)


# ------------ A custom target ------------

CUSTOM_TARGET = ".".join([__name__, "CustomCPU"])


class CustomCPU(CPU):
    """Extend from the CPU target
    """
    pass


class CustomCPUDispatcher(CPUDispatcher):
    pass


target_registry[CUSTOM_TARGET] = CustomCPU
dispatcher_registry[target_registry[CUSTOM_TARGET]] = CustomCPUDispatcher


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
        return TargetConfig.switch_target(self.retarget)

    @contextmanager
    def check_retarget_error(self):
        with self.assertRaises(errors.NumbaError) as raises:
            yield
        self.assertIn(f"{CUSTOM_TARGET} != cpu", str(raises.exception))

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

    def test_case1(self):
        flex_target = self.functions["flex_target"]

        @njit
        def foo(x):
            x = flex_target(x)
            return x

        with self.switch_target():
            r = foo(123)
        self.assertEqual(r, 123 + 1000)

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
