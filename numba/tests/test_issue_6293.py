from numba import jit, types
from numba.core import ir
import unittest

class TestIssue6293(unittest.TestCase):
    def test_issue_6293(self):
        @jit(nopython=True)
        def confuse_typer(x):
            if x == x:
                return int(x)
            else:
                return x

        confuse_typer.compile((types.float64,))
        cres = confuse_typer.overloads[(types.float64,)]
        typemap = cres.type_annotation.typemap
        return_vars = {}

        for block in cres.type_annotation.blocks.values():
            for inst in block.body:
                if isinstance(inst, ir.Return):
                    varname = inst.value.name
                    return_vars[varname] = typemap[varname]

        self.assertTrue(all(vt == types.float64 for vt in return_vars.values()))