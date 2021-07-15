import inspect
import re

from numba.tests.support import TestCase, override_config, needs_subprocess
from numba import jit, njit
from numba.core import types
import unittest
import llvmlite.binding as llvm

#NOTE: These tests are potentially sensitive to changes in SSA or lowering
# behaviour and may need updating should changes be made to the corresponding
# algorithms.


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


class TestDebugInfoEmission(TestCase):
    """ Tests that debug info is emitted correctly.
    """

    _NUMBA_OPT_0_ENV = {'NUMBA_OPT': '0'}

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

    def _subprocess_test_runner(self, test_name):
        themod = self.__module__
        thecls = type(self).__name__
        self.subprocess_test_runner(test_module=themod,
                                    test_class=thecls,
                                    test_name=test_name,
                                    envvars=self._NUMBA_OPT_0_ENV)

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
        if debug metadata is associated with it, it manifests as the
        prologue_end being associated with the end_sequence or similar (due to
        the way code gen works for the entry block)."""

        @njit(debug=True)
        def foo(a):
            return a + 1
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
        # This test relies on the CFG not being simplified as it checks the jump
        # from the entry block to the first basic block. Force OPT as 0, if set
        # via the env var the targetmachine and various pass managers all end up
        # at OPT 0 and the IR is minimally transformed prior to lowering to ELF.
        self._subprocess_test_runner('test_DILocation_entry_blk_impl')

    @needs_subprocess
    def test_DILocation_decref_impl(self):
        """ This tests that decref's generated from `ir.Del`s as variables go
        out of scope do not have debuginfo associated with them (the location of
        `ir.Del` is an implementation detail).
        """

        @njit(debug=True)
        def sink(*x):
            pass

        # This function has many decrefs!
        @njit(debug=True)
        def foo(a):
            x = (a, a)
            if a[0] == 0:
                sink(x)
                return 12
            z = x[0][0]
            return z

        sig = (types.float64[::1],)
        full_ir = self._get_llvmir(foo, sig=sig)

        # make sure decref lines end with `meminfo.<number>)` without !dbg info.
        count = 0
        for line in full_ir.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith('call void @NRT_decref'):
                self.assertRegex(line, r'.*meminfo\.[0-9]+\)$')
                count += 1
        self.assertGreater(count, 0) # make sure there were some decrefs!

    def test_DILocation_decref(self):
        # Test runner for test_DILocation_decref_impl, needs a subprocess
        # with opt=0 to preserve decrefs.
        self._subprocess_test_runner('test_DILocation_decref_impl')

    def test_DILocation_undefined(self):
        """ Tests that DILocation information for undefined vars is associated
        with the line of the function definition (so it ends up in the prologue)
        """
        @njit(debug=True)
        def foo(n):
            if n:
                if n > 0:
                    c = 0
                return c
            else:
                # variable c is not defined in this branch
                c += 1
                return c

        sig = (types.intp,)
        metadata = self._get_metadata(foo, sig=sig)
        pysrc, pysrc_line_start = inspect.getsourcelines(foo)
        # Looks for versions of variable "c" and captures the line number
        expr = r'.*!DILocalVariable\(name: "c\$?[0-9]?",.*line: ([0-9]+),.*'
        matcher = re.compile(expr)
        associated_lines = set()
        for md in metadata:
            match = matcher.match(md)
            if match:
                groups = match.groups()
                self.assertEqual(len(groups), 1)
                associated_lines.add(int(groups[0]))
        self.assertEqual(len(associated_lines), 3) # 3 versions of 'c'
        self.assertIn(pysrc_line_start, associated_lines)

    def test_DILocation_versioned_variables(self):
        """ Tests that DILocation information for versions of variables matches
        up to their definition site."""
        # Note: there's still something wrong in the DI/SSA naming, the ret c is
        # associated with the logically first definition.

        @njit(debug=True)
        def foo(n):
            if n:
                c = 5
            else:
                c = 1
            return c

        sig = (types.intp,)
        metadata = self._get_metadata(foo, sig=sig)
        pysrc, pysrc_line_start = inspect.getsourcelines(foo)

        # Looks for SSA versioned names i.e. <basename>$<version id> of the
        # variable 'c' and captures the line
        expr = r'.*!DILocalVariable\(name: "c\$[0-9]?",.*line: ([0-9]+),.*'
        matcher = re.compile(expr)
        associated_lines = set()
        for md in metadata:
            match = matcher.match(md)
            if match:
                groups = match.groups()
                self.assertEqual(len(groups), 1)
                associated_lines.add(int(groups[0]))
        self.assertEqual(len(associated_lines), 2) # 2 SSA versioned names 'c'

        # Now find the `c = ` lines in the python source
        py_lines = set()
        for ix, pyln in enumerate(pysrc):
            if 'c = ' in pyln:
                py_lines.add(ix + pysrc_line_start)
        self.assertEqual(len(py_lines), 2) # 2 assignments to c

        # check that the DILocation from the DI for `c` matches the python src
        self.assertEqual(associated_lines, py_lines)


if __name__ == '__main__':
    unittest.main()
