import re
import inspect

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

        metadata = self._get_metadata(foo, sig=(types.float64,))
        full_ir = self._get_llvmir(foo, sig=(types.float64,))

        llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

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
        pyln_range = range(pysrc_line_start + 2, pysrc_line_start + len(pysrc))
        for (k, line_no) in zip(sorted(line2dbg), pyln_range):
            dilocation_info = metadata_definition_map[k]
            self.assertIn(f'line: {line_no}', dilocation_info)


if __name__ == '__main__':
    unittest.main()
