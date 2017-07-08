import numba
from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature
from numba import ir_utils, ir
from numba.ir_utils import get_call_table, find_topo_order


def stencil():
    pass

@infer_global(stencil)
class Stencil(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)

class StencilPass(object):
    def __init__(self, func_ir, typemap, calltypes, array_analysis):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.array_analysis = array_analysis

    def run(self):
        call_table, _ = get_call_table(self.func_ir.blocks)
        stencil_calls = []
        for call_varname, call_list in call_table.items():
            if call_list == ['stencil', numba]:
                stencil_calls.append(call_varname)
        if not stencil_calls:
            return  # return early if no stencil calls found
        topo_order = find_topo_order(self.func_ir.blocks)
        stencil_vars = set()
        stencil_ir = []
        for label in reversed(topo_order):
            block = self.func_ir.blocks[label]
            for stmt in reversed(block.body):
                # first find a stencil call
                if (not stencil_vars and isinstance(stmt, ir.Assign)
                        and isinstance(stmt.value, ir.Expr)
                        and stmt.value.op == 'call'
                        and stmt.value.func.name in stencil_calls):
                    assert len(stmt.value.args) == 3
                    stencil_vars.add(stmt.value.args[2].name)
                # collect stencil related IR after call found
                # if stencil_vars:
                #     stmt_vars = set(v.name for v in stmt.list_vars())
                #     if stencil_vars & stmt_vars:
                #         stencil_vars |= stmt_vars
                #         stencil_ir
        return
