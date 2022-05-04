"""
Test bytecode fixes provided in interpreter.py
"""
from numba import njit, objmode
from numba.core.errors import UnsupportedError
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
        param_list = [f"arg{i}" for i in range(n_args + n_kws)]
        args_list = []
        for i in range(n_args + n_kws):
            # Call a function on every 5th argument to ensure
            # we test function calls.
            if i % 5 == 0:
                arg_val = f"pow(arg{i}, 2)"
            else:
                arg_val = f"arg{i}"
            args_list.append(arg_val)
        total_params = ", ".join(param_list)
        func_text = f"def impl({total_params}):\n"
        func_text += "    return sum_jit_func(\n"
        for i in range(n_args):
            func_text += f"        {args_list[i]},\n"
        for i in range(n_args, n_args + n_kws):
            func_text += f"        {param_list[i]}={args_list[i]},\n"
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

    @skip_unless_py10
    def test_large_kws_objmode(self):
        """
        Tests calling an objectmode function with > 15 return values.
        """
        def py_func():
            return (
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            )

        @njit
        def objmode_func():
            """
            Wrapper to call py_func from objmode. This tests
            large kws with objmode. If the definition for the
            call is not properly updated this test will fail.
            """
            with objmode(
                a='int64',
                b='int64',
                c='int64',
                d='int64',
                e='int64',
                f='int64',
                g='int64',
                h='int64',
                i='int64',
                j='int64',
                k='int64',
                l='int64',
                m='int64',
                n='int64',
                o='int64',
                p='int64',
            ):
                (
                    a,
                    b,
                    c,
                    d,
                    e,
                    f,
                    g,
                    h,
                    i,
                    j,
                    k,
                    l,
                    m,
                    n,
                    o,
                    p
                ) = py_func()
            return (
                a
                + b
                + c
                + d
                + e
                + f
                + g
                + h
                + i
                + j
                + k
                + l
                + m
                + n
                + o
                + p
            )

        a = sum(list(py_func()))
        b = objmode_func()
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_large_args_inline_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has inlined controlflow.
        """
        def inline_func(flag):
            return sum_jit_func(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1 if flag else 2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                arg41=1,
            )

        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn(
            'You can resolve this issue by moving the control flow out',
            str(raises.exception)
        )

    @skip_unless_py10
    def test_large_args_noninlined_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """
        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                a_val,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                arg41=1,
            )

        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_all_args_inline_controlflow(self):
        """
        Tests generating only large args when one of the inputs
        has inlined controlflow. This requires a special check
        inside peep_hole_call_function_ex_to_call_function_kw
        because it usually only handles varkwargs.
        """
        def inline_func(flag):
            return sum_jit_func(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1 if flag else 2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )

        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn(
            'You can resolve this issue by moving the control flow out',
            str(raises.exception)
        )

    @skip_unless_py10
    def test_all_args_noninlined_controlflow(self):
        """
        Tests generating large args when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """
        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                a_val,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            )

        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)

    @skip_unless_py10
    def test_large_kws_inline_controlflow(self):
        """
        Tests generating large kws when one of the inputs
        has inlined controlflow.
        """
        def inline_func(flag):
            return sum_jit_func(
                arg0=1,
                arg1=1,
                arg2=1,
                arg3=1,
                arg4=1,
                arg5=1,
                arg6=1,
                arg7=1,
                arg8=1,
                arg9=1,
                arg10=1,
                arg11=1,
                arg12=1,
                arg13=1,
                arg14=1,
                arg15=1 if flag else 2,
            )

        with self.assertRaises(UnsupportedError) as raises:
            njit()(inline_func)(False)
        self.assertIn(
            'You can resolve this issue by moving the control flow out',
            str(raises.exception)
        )

    @skip_unless_py10
    def test_large_kws_noninlined_controlflow(self):
        """
        Tests generating large kws when one of the inputs
        has the change suggested in the error message
        for inlined control flow.
        """
        def inline_func(flag):
            a_val = 1 if flag else 2
            return sum_jit_func(
                arg0=1,
                arg1=1,
                arg2=1,
                arg3=1,
                arg4=1,
                arg5=1,
                arg6=1,
                arg7=1,
                arg8=1,
                arg9=1,
                arg10=1,
                arg11=1,
                arg12=1,
                arg13=1,
                arg14=1,
                arg15=a_val,
            )

        py_func = inline_func
        cfunc = njit()(inline_func)
        a = py_func(False)
        b = cfunc(False)
        self.assertEqual(a, b)
