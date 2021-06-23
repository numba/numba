import inspect
import os
import re
import subprocess
import sys

from numba.tests.support import TestCase, override_config
from numba import jit, njit
from numba.core import types
import unittest
import llvmlite.binding as llvm


class TestDebugInfo(TestCase):
    """
    These tests only checks the compiled assembly for debuginfo.
    """

    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        m = re.search(r"\.section.+debug", asm, re.I)
        got = m is not None
        self.assertEqual(expect, got, msg='debug info not found in:\n%s' % asm)

    def test_no_debuginfo_in_asm(self):
        @jit(nopython=True, debug=False)
        def foo(x):
            return x

        self._check(foo, sig=(types.int32,), expect=False)

    def test_debuginfo_in_asm(self):
        @jit(nopython=True, debug=True)
        def foo(x):
            return x

        self._check(foo, sig=(types.int32,), expect=True)

    def test_environment_override(self):
        with override_config('DEBUGINFO_DEFAULT', 1):
            # Using default value
            @jit(nopython=True)
            def foo(x):
                return x
            self._check(foo, sig=(types.int32,), expect=True)

            # User override default
            @jit(nopython=True, debug=False)
            def bar(x):
                return x
            self._check(bar, sig=(types.int32,), expect=False)


# some tests need clean environments to hide optimisation default states
_exec_cond = os.environ.get('SUBPROC_TEST', None) == '1'
needs_subprocess = unittest.skipUnless(_exec_cond, "needs subprocess harness")


class TestDebugInfoEmission(TestCase):
    """ Tests that debug info is emitted correctly.
    """

    def _get_llvmir(self, fn, sig):
        with override_config('OPT', 0):
            fn.compile(sig)
            return fn.inspect_llvm(sig)

    def _get_metadata(self, fn, sig):
        ll = self._get_llvmir(fn, sig).splitlines()
        meta_re = re.compile(r'![0-9]+ =.*')
        metadata = []
        for line in ll:
            if meta_re.match(line):
                metadata.append(line)
        return metadata

    def test_DW_LANG(self):

        @njit(debug=True)
        def foo():
            pass

        metadata = self._get_metadata(foo, sig=())
        DICompileUnit = metadata[0]
        self.assertEqual('!0', DICompileUnit[:2])
        self.assertIn('!DICompileUnit(language: DW_LANG_C_plus_plus',
                      DICompileUnit)
        self.assertIn('producer: "Numba"', DICompileUnit)

    def test_DILocation(self):
        """ Tests that DILocation information is reasonable.
        """
        @njit(debug=True, error_model='numpy')
        def foo(a):
            b = a + 1.23
            c = a * 2.34
            d = b / c
            print(d)
            return d

        # the above produces LLVM like:
        # define function() {
        # entry:
        #   alloca
        #   store 0 to alloca
        #   <arithmetic for doing the operations on b, c, d>
        #   setup for print
        #   branch
        # other_labels:
        # ... <elided>
        # }
        #
        # The following checks that:
        # * the alloca and store have no !dbg
        # * the arithmetic occurs in the order defined and with !dbg
        # * that the !dbg entries are monotonically increasing in value with
        #   source line number

        sig = (types.float64,)
        metadata = self._get_metadata(foo, sig=sig)
        full_ir = self._get_llvmir(foo, sig=sig)

        module = llvm.parse_assembly(full_ir)

        name = foo.overloads[foo.signatures[0]].fndesc.mangled_name
        funcs = [x for x in module.functions if x.name == name]
        self.assertEqual(len(funcs), 1)
        func = funcs[0]
        blocks = [x for x in func.blocks]
        self.assertGreater(len(blocks), 1)
        block = blocks[0]

        # Find non-call instr and check the sequence is as expected
        instrs = [x for x in block.instructions if x.opcode != 'call']
        op_seq = [x.opcode for x in instrs]
        op_expect = ('fadd', 'fmul', 'fdiv')
        self.assertIn(''.join(op_expect), ''.join(op_seq))

        # Parse out metadata from end of each line, check it monotonically
        # ascends with LLVM source line. Also store all the dbg references,
        # these will be checked later.
        line2dbg = set()
        re_dbg_ref = re.compile(r'.*!dbg (![0-9]+).*$')
        found = -1
        for instr in instrs:
            inst_as_str = str(instr)
            matched = re_dbg_ref.match(inst_as_str)
            if not matched:
                # if there's no match, ensure it is one of alloca or store,
                # it's important that the zero init/alloca instructions have
                # no dbg data
                accepted = ('alloca ', 'store ')
                self.assertTrue(any([x in inst_as_str for x in accepted]))
                continue
            groups = matched.groups()
            self.assertEqual(len(groups), 1)
            dbg_val = groups[0]
            int_dbg_val = int(dbg_val[1:])
            if found >= 0:
                self.assertTrue(int_dbg_val >= found)
            found = int_dbg_val
            # some lines will alias dbg info, this is fine, it's only used to
            # make sure that the line numbers are correct WRT python
            line2dbg.add(dbg_val)

        pysrc, pysrc_line_start = inspect.getsourcelines(foo)

        # build a map of dbg reference to DI* information
        metadata_definition_map = dict()
        meta_definition_split = re.compile(r'(![0-9]+) = (.*)')
        for line in metadata:
            matched = meta_definition_split.match(line)
            if matched:
                dbg_val, info = matched.groups()
                metadata_definition_map[dbg_val] = info

        # Pull out metadata entries referred to by the llvm line end !dbg
        # check they match the python source, the +2 is for the @njit decorator
        # and the function definition line.
        offsets = [0,  # b = a + 1
                   1,  # a * 2.34
                   2,  # d = b / c
                   3,  # print(d)
                   ]
        pyln_range = [pysrc_line_start + 2 + x for x in offsets]

        # do the check
        for (k, line_no) in zip(sorted(line2dbg, key=lambda x: int(x[1:])),
                                pyln_range):
            dilocation_info = metadata_definition_map[k]
            self.assertIn(f'line: {line_no}', dilocation_info)

        # Check that variable "a" is declared as on the same line as function
        # definition.
        expr = r'.*!DILocalVariable\(name: "a",.*line: ([0-9]+),.*'
        match_local_var_a = re.compile(expr)
        for entry in metadata_definition_map.values():
            matched = match_local_var_a.match(entry)
            if matched:
                groups = matched.groups()
                self.assertEqual(len(groups), 1)
                dbg_line = int(groups[0])
                self.assertEqual(dbg_line, pysrc_line_start)
                break
        else:
            self.fail('Assertion on DILocalVariable not made')

    @needs_subprocess
    def test_DILocation_entry_blk_impl(self):
        """ This tests that the unconditional jump emitted at the tail of
        the entry block has no debug metadata associated with it. In practice,
        if this is not the case, it manifests as the prologue_end being
        associated with the end_sequence or similar (due to the way code gen
        works for the entry block)."""

        with override_config('OPT', 0):
            # NOTE: This has to be declared and compiled under the override
            # ctx manager, this is because a codegen is created at decoration
            # point and it will get the default OPT (3) if not declared with the
            # override in place.
            @njit(debug=True)
            def foo(a):
                b = a + 1
                c = b * 2.34
                d = (a, 10 * c)
                return d
            foo(123)

        full_ir = foo.inspect_llvm(foo.signatures[0])
        # The above produces LLVM like:
        #
        # define function() {
        # entry:
        #   alloca
        #   store 0 to alloca
        #   unconditional jump to body:
        #
        # body:
        # ... <elided>
        # }

        module = llvm.parse_assembly(full_ir)
        name = foo.overloads[foo.signatures[0]].fndesc.mangled_name
        funcs = [x for x in module.functions if x.name == name]
        self.assertEqual(len(funcs), 1)
        func = funcs[0]
        blocks = [x for x in func.blocks]
        self.assertEqual(len(blocks), 2)
        entry_block, body_block = blocks

        # Assert that the tail of the entry block is an unconditional jump to
        # the body block and that the jump has no associated debug info.
        entry_instr = [x for x in entry_block.instructions]
        ujmp = entry_instr[-1]
        self.assertEqual(ujmp.opcode, 'br')
        ujmp_operands = [x for x in ujmp.operands]
        self.assertEqual(len(ujmp_operands), 1)
        target_data = ujmp_operands[0]
        target = str(target_data).split(':')[0].strip()
        # check the unconditional jump target is to the body block
        self.assertEqual(target, body_block.name)
        # check the uncondition jump instr itself has no metadata
        self.assertTrue(str(ujmp).endswith(target))

    def test_DILocation_entry_blk(self):
        # Test runner for test_DILocation_entry_blk_impl, needs a subprocess
        # as jitting literally anything at any point in the lifetime of the
        # process ends up with a codegen at opt 3. This is not amenable to this
        # test!
        themod = self.__module__
        thecls = type(self).__name__
        injected_method = f'{themod}.{thecls}.test_DILocation_entry_blk_impl'
        cmd = [sys.executable, '-m', 'numba.runtests', injected_method]
        env_copy = os.environ.copy()
        env_copy['SUBPROC_TEST'] = '1'
        status = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=60, check=True,
                                env=env_copy, universal_newlines=True)
        self.assertEqual(status.returncode, 0)
        self.assertIn('OK', status.stderr       )
        self.assertTrue('FAIL' not in status.stderr)
        self.assertTrue('ERROR' not in status.stderr)


if __name__ == '__main__':
    unittest.main()
