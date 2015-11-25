from __future__ import print_function, division, absolute_import

from collections import defaultdict

from . import config


class Rewrite(object):
    '''Defines the abstract base class for Numba rewrites.
    '''

    def __init__(self, pipeline):
        '''Constructor for the Rewrite class.
        '''
        self.pipeline = pipeline

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
        if not kind in self._kinds:
            raise KeyError("invalid kind %r" % (kind,))
        def do_register(rewrite_cls):
            if not issubclass(rewrite_cls, Rewrite):
                raise TypeError('{0} is not a subclass of Rewrite'.format(
                    rewrite_cls))
            self.rewrites[kind].append(rewrite_cls)
            return rewrite_cls
        return do_register

    def apply(self, kind, pipeline, interp):
        '''Given a pipeline and a dictionary of basic blocks, exhaustively
        attempt to apply all registered rewrites to all basic blocks.
        '''
        assert kind in self._kinds
        blocks = interp.blocks
        old_blocks = blocks.copy()
        for rewrite_cls in self.rewrites[kind]:
            # Exhaustively apply a rewrite until it stops matching.
            rewrite = rewrite_cls(pipeline)
            work_list = list(blocks.items())
            while work_list:
                key, block = work_list.pop()
                matches = rewrite.match(interp, block, pipeline.typemap,
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


from numba import ir

@register_rewrite('before-inference')
class RewriteConstGetitems(Rewrite):
    """
    Rewrite IR expressions of the kind `getitem(value=arr, index=$constXX)`
    where `$constXX` is a known constant as
    `static_getitem(value=arr, index=<constant value>)`.
    """

    def match(self, interp, block, typemap, calltypes):
        self.getitems = getitems = []
        self.block = block
        # Detect all getitem expressions and find which ones can be
        # rewritten
        for inst in block.body:
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr):
                expr = inst.value
                if expr.op == 'getitem':
                    try:
                        defn = interp.get_definition(expr.index)
                    except KeyError:
                        continue
                    try:
                        const = defn.infer_constant()
                    except TypeError:
                        continue
                    getitems.append((expr, const))

        return len(getitems) > 0

    def apply(self):
        """
        Rewrite all matching getitems as static_getitems.
        """
        for expr, const in self.getitems:
            expr.op = 'static_getitem'
            expr.index_var = expr.index
            expr.index = const
        return self.block


@register_rewrite('before-inference')
class RewriteConstSetitems(Rewrite):
    """
    Rewrite IR statements of the kind `setitem(target=arr, index=$constXX, ...)`
    where `$constXX` is a known constant as
    `static_setitem(target=arr, index=<constant value>, ...)`.
    """

    def match(self, interp, block, typemap, calltypes):
        self.setitems = setitems = {}
        self.block = block
        # Detect all setitem statements and find which ones can be
        # rewritten
        for inst in block.body:
            if isinstance(inst, ir.SetItem):
                try:
                    defn = interp.get_definition(inst.index)
                except KeyError:
                    continue
                try:
                    const = defn.infer_constant()
                except TypeError:
                    continue
                setitems[inst] = const

        return len(setitems) > 0

    def apply(self):
        """
        Rewrite all matching getitems as static_getitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.setitems:
                const = self.setitems[inst]
                new_inst = ir.StaticSetItem(inst.target, const,
                                            inst.index, inst.value, inst.loc)
                new_block.append(new_inst)
            else:
                new_block.append(inst)
        return new_block
