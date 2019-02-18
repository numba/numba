# Tests numba.analysis functions
from __future__ import print_function, absolute_import, division

from numba.compiler import compile_isolated, run_frontend
from numba import types, rewrites, ir
from .support import TestCase, MemoryLeakMixin


from numba.analysis import dead_branch_prune


def compile_to_ir(func):
    func_ir = run_frontend(func)

    class MockPipeline(object):
        def __init__(self, func_ir):
            self.typingctx = None
            self.targetctx = None
            self.args = None
            self.func_ir = func_ir
            self.typemap = None
            self.return_type = None
            self.calltypes = None
    # call this to get print etc rewrites
    rewrites.rewrite_registry.apply('before-inference', MockPipeline(func_ir),
                                    func_ir)
    return func_ir


class TestBranchPrune(MemoryLeakMixin, TestCase):
    """
    Tests branch pruning
    """
    _DEBUG = False

    def assert_prune(self, func, args_tys, prune, *args):
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

        func_ir = compile_to_ir(func)
        before = func_ir.copy()
        if self._DEBUG:
            print("=" * 80)
            print("before prune")
            func_ir.dump()

        dead_branch_prune(func_ir, args_tys)

        after = func_ir
        if self._DEBUG:
            print("after prune")
            func_ir.dump()

        # find *all* branches
        def find_branches(the_ir):
            branches = []
            for blk in the_ir.blocks.values():
                tmp = [_ for _ in blk.find_insts(cls=ir.Branch)]
                branches.extend(tmp)
            return branches

        before_branches = find_branches(before)
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
            print("new_labels", new_labels)
            print("original_labels", original_labels)
            print("expect_removed", expect_removed)
            raise e

        cres = compile_isolated(func, args_tys)
        res = cres.entry_point(*args)
        expected = func(*args)
        self.assertEqual(res, expected)

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
