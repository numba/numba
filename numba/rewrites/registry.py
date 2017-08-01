from __future__ import print_function, division, absolute_import

from collections import defaultdict

from numba import config


class Rewrite(object):
    '''Defines the abstract base class for Numba rewrites.
    '''

    def __init__(self, pipeline):
        '''Constructor for the Rewrite class.
        '''
        self.pipeline = pipeline

    def match(self, func_ir, block, typemap, calltypes):
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
    _kinds = frozenset(['before-inference', 'after-inference'])

    def __init__(self):
        '''Constructor for the rewrite registry.  Initializes the rewrites
        member to an empty list.
        '''
        self.rewrites = defaultdict(list)

    def register(self, kind):
        """
        Decorator adding a subclass of Rewrite to the registry for
        the given *kind*.
        """
        if kind not in self._kinds:
            raise KeyError("invalid kind %r" % (kind,))
        def do_register(rewrite_cls):
            if not issubclass(rewrite_cls, Rewrite):
                raise TypeError('{0} is not a subclass of Rewrite'.format(
                    rewrite_cls))
            self.rewrites[kind].append(rewrite_cls)
            return rewrite_cls
        return do_register

    def apply(self, kind, pipeline, func_ir):
        '''Given a pipeline and a dictionary of basic blocks, exhaustively
        attempt to apply all registered rewrites to all basic blocks.
        '''
        assert kind in self._kinds
        blocks = func_ir.blocks
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites[kind]:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls(pipeline)
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(func_ir, block, pipeline.typemap,
                                        pipeline.calltypes)
                if matches:
                    if config.DEBUG or config.DUMP_IR:
                        print("_" * 70)
                        print("REWRITING (%s):" % rewrite_cls.__name__)
                        block.dump()
                        print("_" * 60)
                    new_block = rewrite.apply()
                    blocks[key] = new_block
                    work_list.append((key, new_block))
                    if config.DEBUG or config.DUMP_IR:
                        new_block.dump()
                        print("_" * 70)
        # If any blocks were changed, perform a sanity check.
        for key, block in blocks.items():
            if block != old_blocks[key]:
                block.verify()


rewrite_registry = RewriteRegistry()
register_rewrite = rewrite_registry.register
