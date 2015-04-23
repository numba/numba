from __future__ import print_function, division, absolute_import

from . import ir, types

class Rewrite(object):
    def match(self, block, typemap, calltypes):
        return False

    def apply(self, block):
        raise NotImplementedError("Abstract Rewrite.apply() called!")


class RewriteRegistry(object):
    def __init__(self):
        self.rewrites = []

    def register(self, rewrite_cls):
        assert issubclass(rewrite_cls, Rewrite)
        self.rewrites.append(rewrite_cls)

    def apply(self, blocks, typemap, calltypes):
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls()
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(block, typemap, calltypes)
                if matches:
                    new_block = rewrite.apply(block)
                    blocks[key] = new_block
                    work_list.append((key, new_block))
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register


@register_rewrite
class RewriteArrayExprs(Rewrite):
    def match(self, block, typemap, calltypes):
        result = False
        cands = set()
        matches = set()
        for instr in block.body:
            is_cand = (isinstance(instr, ir.Assign) and
                       isinstance(instr.value, ir.Expr) and
                       instr.value.op == 'binop' and
                       instr.value.fn in []) #FIXME
            if is_cand:
                raise NotImplementedError("Development frontier.")
        return result

    def rewrite(self, block):
        raise NotImplementedError("Development frontier.")
