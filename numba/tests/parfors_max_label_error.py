"""
This is a test for an error with ir_utils._max_label not being updated
correctly. As a result of the error, inline_closurecall will incorrectly
overwrite an existing label, resulting in code that creates an effect-free
infinite loop, which is an undefined behavior to LLVM. LLVM will then assume
the function can never execute and an alternative code path will be taken
in spite of the value of any conditional branch that is guarding the alternative
code path.
"""
import numpy as np

from numba import njit
from numba.core.dispatcher import TargetConfigurationStack
from numba.core.target_extension import (
    dispatcher_registry,
    CPUDispatcher,
    CPU,
    target_registry,
)
from numba.core.retarget import BasicRetarget

# ------------ A "CustomCPU" target ------------


class CustomCPU(CPU):
    pass


class CustomCPUDispatcher(CPUDispatcher):
    pass


target_registry["CustomCPU"] = CustomCPU
dispatcher_registry[target_registry["CustomCPU"]] = CustomCPUDispatcher


# ------------ For switching target ------------


class MyCustomTarget(BasicRetarget):
    @property
    def output_target(self):
        return "CustomCPU"

    def compile_retarget(self, cpu_disp):
        kernel = njit(_target="CustomCPU", parallel=True)(cpu_disp.py_func)
        return kernel


retarget = MyCustomTarget()


# ------------ Functions being tested ------------
@njit
def f(a):
    return np.arange(a.size)


def main():
    a = np.ones(20)
    with TargetConfigurationStack.switch_target(retarget):
        r = f(a)
    np.testing.assert_equal(r, f.py_func(a))
    print("TEST PASSED")


if __name__ == "__main__":
    main()
