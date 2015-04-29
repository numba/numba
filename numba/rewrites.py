from __future__ import print_function, division, absolute_import


class Rewrite(object):
    def match(self, block, typemap, calltypes):
        return False

    def apply(self):
        raise NotImplementedError("Abstract Rewrite.apply() called!")


class RewriteRegistry(object):
    def __init__(self):
        self.rewrites = []

    def register(self, rewrite_cls):
        assert issubclass(rewrite_cls, Rewrite)
        self.rewrites.append(rewrite_cls)
        return rewrite_cls

    def apply(self, pipeline, blocks):
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls(pipeline)
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(block, pipeline.typemap,
                                        pipeline.calltypes)
                if matches:
                    new_block = rewrite.apply()
                    blocks[key] = new_block
                    work_list.append((key, new_block))
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register
