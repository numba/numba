# Tests numba.analysis functions
import collections
import types as pytypes

import numpy as np
from numba.core.compiler import compile_isolated, run_frontend, Flags, StateDict
from numba import jit, njit
from numba.core import types, errors, ir, rewrites, ir_utils, utils
from numba.tests.support import TestCase, MemoryLeakMixin, SerialMixin

from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants

_GLOBAL = 123

enable_pyobj_flags = Flags()
enable_pyobj_flags.set("enable_pyobject")


def compile_to_ir(func):
    func_ir = run_frontend(func)
    state = StateDict()
    state.func_ir = func_ir
    state.typemap = None
    state.calltypes = None

    # call this to get print etc rewrites
    rewrites.rewrite_registry.apply('before-inference', state)
    return func_ir


class TestBranchPruneBase(MemoryLeakMixin, TestCase):
    """
    Tests branch pruning
    """
    _DEBUG = False

    # find *all* branches
    def find_branches(self, the_ir):
        branches = []
        for blk in the_ir.blocks.values():
            tmp = [_ for _ in blk.find_insts(cls=ir.Branch)]
            branches.extend(tmp)
        return branches

    def assert_prune(self, func, args_tys, prune, *args, **kwargs):
        # This checks that the expected pruned branches have indeed been pruned.
        # func is a python function to assess
        # args_tys is the numba types arguments tuple
        # prune arg is a list, one entry per branch. The value in the entry is
        # encoded as follows:
        # True: using constant inference only, the True branch will be pruned
        # False: using constant inference only, the False branch will be pruned
        # None: under no circumstances should this branch be pruned
        # *args: the argument instances to pass to the function to check
        #        execution is still valid post transform
        # **kwargs:
        #        - flags: compiler.Flags instance to pass to `compile_isolated`,
        #          permits use of e.g. object mode

        func_ir = compile_to_ir(func)
        before = func_ir.copy()
        if self._DEBUG:
            print("=" * 80)
            print("before prune")
            func_ir.dump()

        rewrite_semantic_constants(func_ir, args_tys)
        dead_branch_prune(func_ir, args_tys)

        after = func_ir
        if self._DEBUG:
            print("after prune")
            func_ir.dump()

        before_branches = self.find_branches(before)
        self.assertEqual(len(before_branches), len(prune))

        # what is expected to be pruned
        expect_removed = []
        for idx, prune in enumerate(prune):
            branch = before_branches[idx]
            if prune is True:
                expect_removed.append(branch.truebr)
            elif prune is False:
                expect_removed.append(branch.falsebr)
            elif prune is None:
                pass  # nothing should be removed!
            elif prune == 'both':
                expect_removed.append(branch.falsebr)
                expect_removed.append(branch.truebr)
            else:
                assert 0, "unreachable"

        # compare labels
        original_labels = set([_ for _ in before.blocks.keys()])
        new_labels = set([_ for _ in after.blocks.keys()])
        # assert that the new labels are precisely the original less the
        # expected pruned labels
        try:
            self.assertEqual(new_labels, original_labels - set(expect_removed))
        except AssertionError as e:
            print("new_labels", sorted(new_labels))
            print("original_labels", sorted(original_labels))
            print("expect_removed", sorted(expect_removed))
            raise e

        supplied_flags = kwargs.pop('flags', False)
        compiler_kws = {'flags': supplied_flags} if supplied_flags else {}
        cres = compile_isolated(func, args_tys, **compiler_kws)
        if args is None:
            res = cres.entry_point()
            expected = func()
        else:
            res = cres.entry_point(*args)
            expected = func(*args)
        self.assertEqual(res, expected)


class TestBranchPrune(TestBranchPruneBase, SerialMixin):

    def test_single_if(self):

        def impl(x):
            if 1 == 0:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [True], None)

        def impl(x):
            if 1 == 1:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [False], None)

        def impl(x):
            if x is None:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [False], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

        def impl(x):
            if x == 10:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

        def impl(x):
            if x == 10:
                z = 3.14159  # noqa: F841 # no effect

        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

        # TODO: cannot handle this without const prop
        # def impl(x):
        #     z = None
        #     y = z
        #     if x == y:
        #         print("x is 10")

        # self.assert_prune(impl, (types.NoneType('none'),), [None], None)
        # self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)

    def test_single_if_else(self):

        def impl(x):
            if x is None:
                return 3.14159
            else:
                return 1.61803

        self.assert_prune(impl, (types.NoneType('none'),), [False], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True], 10)

    def test_single_if_const_val(self):

        def impl(x):
            if x == 100:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

        def impl(x):
            # switch the condition order
            if 100 == x:
                return 3.14159

        self.assert_prune(impl, (types.NoneType('none'),), [True], None)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def test_single_if_else_two_const_val(self):

        def impl(x, y):
            if x == y:
                return 3.14159
            else:
                return 1.61803

        self.assert_prune(impl, (types.IntegerLiteral(100),) * 2, [None], 100,
                          100)
        self.assert_prune(impl, (types.NoneType('none'),) * 2, [False], None,
                          None)
        self.assert_prune(impl, (types.IntegerLiteral(100),
                                 types.NoneType('none'),), [True], 100, None)
        self.assert_prune(impl, (types.IntegerLiteral(100),
                                 types.IntegerLiteral(1000)), [None], 100, 1000)

    def test_single_if_else_w_following_undetermined(self):

        def impl(x):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                dead = 7  # noqa: F841 # no effect

            if x_is_none_work:
                y = 10
            else:
                y = -3
            return y

        self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

        def impl(x):
            x_is_none_work = False
            if x is None:
                x_is_none_work = True
            else:
                pass  # force the True branch exit to be on backbone

            if x_is_none_work:
                y = 10
            else:
                y = -3
            return y

        self.assert_prune(impl, (types.NoneType('none'),), [None, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def test_double_if_else_rt_const(self):

        def impl(x):
            one_hundred = 100
            x_is_none_work = 4
            if x is None:
                x_is_none_work = 100
            else:
                dead = 7  # noqa: F841 # no effect

            if x_is_none_work == one_hundred:
                y = 10
            else:
                y = -3

            return y, x_is_none_work

        self.assert_prune(impl, (types.NoneType('none'),), [False, None], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, None], 10)

    def test_double_if_else_non_literal_const(self):

        def impl(x):
            one_hundred = 100
            if x == one_hundred:
                y = 3.14159
            else:
                y = 1.61803
            return y

        # no prune as compilation specialization on literal value not permitted
        self.assert_prune(impl, (types.IntegerLiteral(10),), [None], 10)
        self.assert_prune(impl, (types.IntegerLiteral(100),), [None], 100)

    def test_single_two_branches_same_cond(self):

        def impl(x):
            if x is None:
                y = 10
            else:
                y = 40

            if x is not None:
                z = 100
            else:
                z = 400

            return z, y

        self.assert_prune(impl, (types.NoneType('none'),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_none(self):

        def impl(x=None):
            if x is None:
                y = 10
            else:
                y = 40

            if x is not None:
                z = 100
            else:
                z = 400

            return z, y

        self.assert_prune(impl, (types.Omitted(None),),
                          [False, True], None)
        self.assert_prune(impl, (types.NoneType('none'),), [False, True], None)
        self.assert_prune(impl, (types.IntegerLiteral(10),), [True, False], 10)

    def test_cond_is_kwarg_value(self):

        def impl(x=1000):
            if x == 1000:
                y = 10
            else:
                y = 40

            if x != 1000:
                z = 100
            else:
                z = 400

            return z, y

        self.assert_prune(impl, (types.Omitted(1000),), [None, None], 1000)
        self.assert_prune(impl, (types.IntegerLiteral(1000),), [None, None],
                          1000)
        self.assert_prune(impl, (types.IntegerLiteral(0),), [None, None], 0)
        self.assert_prune(impl, (types.NoneType('none'),), [True, False], None)

    def test_cond_rewrite_is_correct(self):
        # this checks that when a condition is replaced, it is replace by a
        # true/false bit that correctly represents the evaluated condition
        def fn(x):
            if x is None:
                return 10
            return 12

        def check(func, arg_tys, bit_val):
            func_ir = compile_to_ir(func)

            # check there is 1 branch
            before_branches = self.find_branches(func_ir)
            self.assertEqual(len(before_branches), 1)

            # check the condition in the branch is a binop
            condition_var = before_branches[0].cond
            condition_defn = ir_utils.get_definition(func_ir, condition_var)
            self.assertEqual(condition_defn.op, 'binop')

            # do the prune, this should kill the dead branch and rewrite the
            #'condition to a true/false const bit
            if self._DEBUG:
                print("=" * 80)
                print("before prune")
                func_ir.dump()
            dead_branch_prune(func_ir, arg_tys)
            if self._DEBUG:
                print("=" * 80)
                print("after prune")
                func_ir.dump()

            # after mutation, the condition should be a const value `bit_val`
            new_condition_defn = ir_utils.get_definition(func_ir, condition_var)
            self.assertTrue(isinstance(new_condition_defn, ir.Const))
            self.assertEqual(new_condition_defn.value, bit_val)

        check(fn, (types.NoneType('none'),), 1)
        check(fn, (types.IntegerLiteral(10),), 0)

    def test_obj_mode_fallback(self):
        # see issue #3879, this checks that object mode fall back doesn't suffer
        # from the IR mutation

        @jit
        def bug(a, b):
            if a.ndim == 1:
                if b is None:
                    return dict()
                return 12
            return []

        self.assertEqual(bug(np.zeros(10), 4), 12)
        self.assertEqual(bug(np.arange(10), None), dict())
        self.assertEqual(bug(np.arange(10).reshape((2, 5)), 10), [])
        self.assertEqual(bug(np.arange(10).reshape((2, 5)), None), [])
        self.assertFalse(bug.nopython_signatures)

    def test_global_bake_in(self):

        def impl(x):
            if _GLOBAL == 123:
                return x
            else:
                return x + 10

        self.assert_prune(impl, (types.IntegerLiteral(1),), [False], 1)

        global _GLOBAL
        tmp = _GLOBAL

        try:
            _GLOBAL = 5

            def impl(x):
                if _GLOBAL == 123:
                    return x
                else:
                    return x + 10

            self.assert_prune(impl, (types.IntegerLiteral(1),), [True], 1)
        finally:
            _GLOBAL = tmp

    def test_freevar_bake_in(self):

        _FREEVAR = 123

        def impl(x):
            if _FREEVAR == 123:
                return x
            else:
                return x + 10

        self.assert_prune(impl, (types.IntegerLiteral(1),), [False], 1)

        _FREEVAR = 12

        def impl(x):
            if _FREEVAR == 123:
                return x
            else:
                return x + 10

        self.assert_prune(impl, (types.IntegerLiteral(1),), [True], 1)

    def test_redefined_variables_are_not_considered_in_prune(self):
        # see issue #4163, checks that if a variable that is an argument is
        # redefined in the user code it is not considered const

        def impl(array, a=None):
            if a is None:
                a = 0
            if a < 0:
                return 10
            return 30

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.NoneType('none'),),
                          [None, None],
                          np.zeros((2, 3)), None)

    def test_comparison_operators(self):
        # see issue #4163, checks that a variable that is an argument and has
        # value None survives TypeError from invalid comparison which should be
        # dead

        def impl(array, a=None):
            x = 0
            if a is None:
                return 10 # dynamic exec would return here
            # static analysis requires that this is executed with a=None,
            # hence TypeError
            if a < 0:
                return 20
            return x

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.NoneType('none'),),
                          [False, 'both'],
                          np.zeros((2, 3)), None)

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.float64,),
                          [None, None],
                          np.zeros((2, 3)), 12.)

    def test_redefinition_analysis_same_block(self):
        # checks that a redefinition in a block with prunable potential doesn't
        # break

        def impl(array, x, a=None):
            b = 0
            if x < 4:
                b = 12
            if a is None:
                a = 0
            else:
                b = 12
            if a < 0:
                return 10
            return 30 + b + a

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.float64, types.NoneType('none'),),
                          [None, None, None],
                          np.zeros((2, 3)), 1., None)

    def test_redefinition_analysis_different_block_can_exec(self):
        # checks that a redefinition in a block that may be executed prevents
        # pruning

        def impl(array, x, a=None):
            b = 0
            if x > 5:
                a = 11 # a redefined, cannot tell statically if this will exec
            if x < 4:
                b = 12
            if a is None: # cannot prune, cannot determine if re-defn occurred
                b += 5
            else:
                b += 7
                if a < 0:
                    return 10
            return 30 + b

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.float64, types.NoneType('none'),),
                          [None, None, None, None],
                          np.zeros((2, 3)), 1., None)

    def test_redefinition_analysis_different_block_cannot_exec(self):
        # checks that a redefinition in a block guarded by something that
        # has prune potential

        def impl(array, x=None, a=None):
            b = 0
            if x is not None:
                a = 11
            if a is None:
                b += 5
            else:
                b += 7
            return 30 + b

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.NoneType('none'), types.NoneType('none')),
                          [True, None],
                          np.zeros((2, 3)), None, None)

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.NoneType('none'), types.float64),
                          [True, None],
                          np.zeros((2, 3)), None, 1.2)

        self.assert_prune(impl,
                          (types.Array(types.float64, 2, 'C'),
                           types.float64, types.NoneType('none')),
                          [None, None],
                          np.zeros((2, 3)), 1.2, None)


class TestBranchPrunePredicates(TestBranchPruneBase, SerialMixin):
    # Really important thing to remember... the branch on predicates end up as
    # POP_JUMP_IF_<bool> and the targets are backwards compared to normal, i.e.
    # the true condition is far jump and the false the near i.e. `if x` would
    # end up in Numba IR as e.g. `branch x 10, 6`.

    _TRUTHY = (1, "String", True, 7.4, 3j)
    _FALSEY = (0, "", False, 0.0, 0j, None)

    def _literal_const_sample_generator(self, pyfunc, consts):
        """
        This takes a python function, pyfunc, and manipulates its co_const
        __code__ member to create a new function with different co_consts as
        supplied in argument consts.

        consts is a dict {index: value} of co_const tuple index to constant
        value used to update a pyfunc clone's co_const.
        """
        pyfunc_code = pyfunc.__code__

        # translate consts spec to update the constants
        co_consts = {k: v for k, v in enumerate(pyfunc_code.co_consts)}
        for k, v in consts.items():
            co_consts[k] = v
        new_consts = tuple([v for _, v in sorted(co_consts.items())])

        # create new code parts
        co_args = [pyfunc_code.co_argcount]

        if utils.PYVERSION >= (3, 8):
            co_args.append(pyfunc_code.co_posonlyargcount)
        co_args.append(pyfunc_code.co_kwonlyargcount)
        co_args.extend([pyfunc_code.co_nlocals,
                        pyfunc_code.co_stacksize,
                        pyfunc_code.co_flags,
                        pyfunc_code.co_code,
                        new_consts,
                        pyfunc_code.co_names,
                        pyfunc_code.co_varnames,
                        pyfunc_code.co_filename,
                        pyfunc_code.co_name,
                        pyfunc_code.co_firstlineno,
                        pyfunc_code.co_lnotab,
                        pyfunc_code.co_freevars,
                        pyfunc_code.co_cellvars
                        ])

        # create code object with mutation
        new_code = pytypes.CodeType(*co_args)

        # get function
        return pytypes.FunctionType(new_code, globals())

    def test_literal_const_code_gen(self):
        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159
            else:
                _CONST2 = "PLACEHOLDER2"
            return _CONST2 + 4

        new = self._literal_const_sample_generator(impl, {1:0, 3:20})
        iconst = impl.__code__.co_consts
        nconst = new.__code__.co_consts
        self.assertEqual(iconst, (None, "PLACEHOLDER1", 3.14159,
                                  "PLACEHOLDER2", 4))
        self.assertEqual(nconst, (None, 0, 3.14159,  20, 4))
        self.assertEqual(impl(None), 3.14159)
        self.assertEqual(new(None), 24)

    def test_single_if_const(self):

        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_negate_const(self):

        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if not _CONST1:
                return 3.14159

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_const(self):

        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if _CONST1:
                return 3.14159
            else:
                return 1.61803

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_negate_const(self):

        def impl(x):
            _CONST1 = "PLACEHOLDER1"
            if not _CONST1:
                return 3.14159
            else:
                return 1.61803

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if const:
                        return 3.14159, const
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_negate_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if not const:
                        return 3.14159, const
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if const:
                        return 3.14159, const
                    else:
                        return 1.61803, const
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_negate_freevar(self):
        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for const in c_inp:

                def func(x):
                    if not const:
                        return 3.14159, const
                    else:
                        return 1.61803, const
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    # globals in this section have absurd names after their test usecase names
    # so as to prevent collisions and permit tests to run in parallel
    def test_single_if_global(self):
        global c_test_single_if_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_global = c

                def func(x):
                    if c_test_single_if_global:
                        return 3.14159, c_test_single_if_global

                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_negate_global(self):
        global c_test_single_if_negate_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_negate_global = c

                def func(x):
                    if c_test_single_if_negate_global:
                        return 3.14159, c_test_single_if_negate_global

                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_global(self):
        global c_test_single_if_else_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_else_global = c

                def func(x):
                    if c_test_single_if_else_global:
                        return 3.14159, c_test_single_if_else_global
                    else:
                        return 1.61803, c_test_single_if_else_global
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_single_if_else_negate_global(self):
        global c_test_single_if_else_negate_global

        for c_inp, prune in (self._TRUTHY, False), (self._FALSEY, True):
            for c in c_inp:
                c_test_single_if_else_negate_global = c

                def func(x):
                    if not c_test_single_if_else_negate_global:
                        return 3.14159, c_test_single_if_else_negate_global
                    else:
                        return 1.61803, c_test_single_if_else_negate_global
                self.assert_prune(func, (types.NoneType('none'),), [prune],
                                  None)

    def test_issue_5618(self):

        @njit
        def foo():
            values = np.zeros(1)
            tmp = 666
            if tmp:
                values[0] = tmp
            return values

        self.assertPreciseEqual(foo.py_func()[0], 666.)
        self.assertPreciseEqual(foo()[0], 666.)


class TestBranchPrunePostSemanticConstRewrites(TestBranchPruneBase):
    # Tests that semantic constants rewriting works by virtue of branch pruning

    def test_array_ndim_attr(self):

        def impl(array):
            if array.ndim == 2:
                if array.shape[1] == 2:
                    return 1
            else:
                return 10

        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'),), [False,
                                                                        None],
                          np.zeros((2, 3)))
        self.assert_prune(impl, (types.Array(types.float64, 1, 'C'),), [True,
                                                                        'both'],
                          np.zeros((2,)))

    def test_tuple_len(self):

        def impl(tup):
            if len(tup) == 3:
                if tup[2] == 2:
                    return 1
            else:
                return 0

        self.assert_prune(impl, (types.UniTuple(types.int64, 3),), [False,
                                                                    None],
                          tuple([1, 2, 3]))
        self.assert_prune(impl, (types.UniTuple(types.int64, 2),), [True,
                                                                    'both'],
                          tuple([1, 2]))

    def test_attr_not_len(self):
        # The purpose of this test is to make sure that the conditions guarding
        # the rewrite part do not themselves raise exceptions.
        # This produces an `ir.Expr` call node for `float.as_integer_ratio`,
        # which is a getattr() on `float`.

        @njit
        def test():
            float.as_integer_ratio(1.23)

        # this should raise a TypingError
        with self.assertRaises(errors.TypingError) as e:
            test()

        self.assertIn("Unknown attribute 'as_integer_ratio'", str(e.exception))

    def test_ndim_not_on_array(self):

        FakeArray = collections.namedtuple('FakeArray', ['ndim'])
        fa = FakeArray(ndim=2)

        def impl(fa):
            if fa.ndim == 2:
                return fa.ndim
            else:
                object()

        # check prune works for array ndim
        self.assert_prune(impl, (types.Array(types.float64, 2, 'C'),), [False],
                          np.zeros((2, 3)))

        # check prune fails for something with `ndim` attr that is not array
        FakeArrayType = types.NamedUniTuple(types.int64, 1, FakeArray)
        self.assert_prune(impl, (FakeArrayType,), [None], fa,
                          flags=enable_pyobj_flags)

    def test_semantic_const_propagates_before_static_rewrites(self):
        # see issue #5015, the ndim needs writing in as a const before
        # the rewrite passes run to make e.g. getitems static where possible
        @njit
        def impl(a, b):
            return a.shape[:b.ndim]

        args = (np.zeros((5, 4, 3, 2)), np.zeros((1, 1)))

        self.assertPreciseEqual(impl(*args), impl.py_func(*args))
