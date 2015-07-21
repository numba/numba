from __future__ import print_function, division, absolute_import

from . import config

class Rewrite(object):
    '''Defines the abstract base class for Numba rewrites.
    '''

    def __init__(self, pipeline, *args, **kws):
        '''Constructor for the Rewrite class.  Stashes any construction
        arguments into attributes of the same name.
        '''
        self.pipeline = pipeline
        self.args = args
        self.kws = kws

    def match(self, block, typemap, calltypes):
        '''Overload this method to check an IR block for matching terms in the
        rewrite.
        '''
        return False

    def apply(self):
        '''Overload this method to return a rewritten IR basic block when a
        match has been found.
        '''
        raise NotImplementedError("Abstract Rewrite.apply() called!")


class RewriteRegistry(object):
    '''Defines a registry for Numba rewrites.
    '''

    def __init__(self):
        '''Constructor for the rewrite registry.  Initializes the rewrites
        member to an empty list.
        '''
        self.rewrites = []

    def register(self, rewrite_cls):
        '''Add a subclass of Rewrite to the registry.
        '''
        if not issubclass(rewrite_cls, Rewrite):
            raise TypeError('{0} is not a subclass of Rewrite'.format(
                rewrite_cls))
        self.rewrites.append(rewrite_cls)
        return rewrite_cls

    def apply(self, pipeline, blocks):
        '''Given a pipeline and a dictionary of basic blocks, exhaustively
        attempt to apply all registered rewrites to all basic blocks.
        '''
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
                    if config.DUMP_IR:
                        print("_" * 70)
                        print("REWRITING:")
                        block.dump()
                        print("_" * 60)
                    new_block = rewrite.apply()
                    blocks[key] = new_block
                    work_list.append((key, new_block))
                    if config.DUMP_IR:
                        new_block.dump()
                        print("_" * 70)
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register
