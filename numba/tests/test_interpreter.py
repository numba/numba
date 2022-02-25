"""
Test bytecode fixes provided in interpreter.py
"""
from numba import njit
from numba.tests.support import TestCase, MemoryLeakMixin, skip_unless_py10


@njit
def sum_jit_func(
    arg0=0,
    arg1=0,
    arg2=0,
    arg3=0,
    arg4=0,
    arg5=0,
    arg6=0,
    arg7=0,
    arg8=0,
    arg9=0,
    arg10=0,
    arg11=0,
    arg12=0,
    arg13=0,
    arg14=0,
    arg15=0,
    arg16=0,
    arg17=0,
    arg18=0,
    arg19=0,
    arg20=0,
    arg21=0,
    arg22=0,
    arg23=0,
    arg24=0,
    arg25=0,
    arg26=0,
    arg27=0,
    arg28=0,
    arg29=0,
    arg30=0,
    arg31=0,
    arg32=0,
    arg33=0,
    arg34=0,
    arg35=0,
    arg36=0,
    arg37=0,
    arg38=0,
    arg39=0,
    arg40=0,
    arg41=0,
    arg42=0,
    arg43=0,
    arg44=0,
    arg45=0,
    arg46=0,
):
    return (
        arg0
        + arg1
        + arg2
        + arg3
        + arg4
        + arg5
        + arg6
        + arg7
        + arg8
        + arg9
        + arg10
        + arg11
        + arg12
        + arg13
        + arg14
        + arg15
        + arg16
        + arg17
        + arg18
        + arg19
        + arg20
        + arg21
        + arg22
        + arg23
        + arg24
        + arg25
        + arg26
        + arg27
        + arg28
        + arg29
        + arg30
        + arg31
        + arg32
        + arg33
        + arg34
        + arg35
        + arg36
        + arg37
        + arg38
        + arg39
        + arg40
        + arg41
        + arg42
        + arg43
        + arg44
        + arg45
        + arg46
    )


class TestCallFunctionExPeepHole(TestCase, MemoryLeakMixin):
    """
    gh #7812

    Tests that check a peephole optimization for Function calls
    in Python 3.10. The bytecode changes when
    (n_args / 2) + n_kws > 15, which moves the arguments from
    the stack into a tuple and dictionary.

    This peephole optimization updates the IR to use the original format.
    There are different paths when n_args > 30 and n_args <= 30 and when
    n_kws > 15 and n_kws <= 15.
    """
    THRESHOLD_ARGS = 31
    THRESHOLD_KWS = 16

    def gen_func(self, n_args, n_kws):
        """
            Generates a function that calls sum_jit_func
            with the desired number of args and kws.
        """
        args_list = [f"arg{i}" for i in range(n_args + n_kws)]
        total_args = ", ".join(args_list)
        func_text = f"def impl({total_args}):\n"
        func_text += "    return sum_jit_func(\n"
        for i in range(n_args):
            func_text += f"        {args_list[i]},\n"
        for i in range(n_args, n_args + n_kws):
            func_text += f"        {args_list[i]}={args_list[i]},\n"
        func_text += "    )\n"
        local_vars = {}
        exec(func_text, {"sum_jit_func": sum_jit_func}, local_vars)
        return local_vars["impl"]

    @skip_unless_py10
    def test_all_args(self):
        """
        Tests calling a function when n_args > 30 and
        n_kws = 0. This shouldn't use the peephole, but
        it should still succeed.
        """
        total_args = [i for i in range(self.THRESHOLD_ARGS)]
        f = self.gen_func(self.THRESHOLD_ARGS, 0)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_all_kws(self):
        """
        Tests calling a function when n_kws > 15 and
        n_args = 0.
        """
        total_args = [i for i in range(self.THRESHOLD_KWS)]
        f = self.gen_func(0, self.THRESHOLD_KWS)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_small_args_small_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args <= 30 and n_kws <= 15
        """
        used_args = self.THRESHOLD_ARGS - 1
        used_kws = self.THRESHOLD_KWS - 1
        total_args = [i for i in range((used_args) + (used_kws))]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_small_args_large_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args <= 30 and n_kws > 15
        """
        used_args = self.THRESHOLD_ARGS - 1
        used_kws = self.THRESHOLD_KWS
        total_args = [i for i in range((used_args) + (used_kws))]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_large_args_small_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args > 30 and n_kws <= 15
        """
        used_args = self.THRESHOLD_ARGS
        used_kws = self.THRESHOLD_KWS - 1
        total_args = [i for i in range((used_args) + (used_kws))]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_large_args_large_kws(self):
        """
        Tests calling a function when (n_args / 2) + n_kws > 15,
        but n_args > 30 and n_kws > 15
        """
        used_args = self.THRESHOLD_ARGS
        used_kws = self.THRESHOLD_KWS
        total_args = [i for i in range((used_args) + (used_kws))]
        f = self.gen_func(used_args, used_kws)
        py_func = f
        cfunc = njit()(f)
        a = py_func(*total_args)
        b = cfunc(*total_args)
        self.assertEqual(a, b)
