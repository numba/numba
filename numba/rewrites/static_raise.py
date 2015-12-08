from numba import ir
from . import register_rewrite, Rewrite


@register_rewrite('before-inference')
class RewriteConstRaises(Rewrite):
    """
    Rewrite IR statements of the kind `raise(value)`
    where `value` is the result of instantiating an exception with
    constant arguments
    into `static_raise(exception_type, constant args)`.

    This allows lowering in nopython mode, where one can't instantiate
    exception instances from runtime data.
    """

    def _is_exception_type(self, const):
        return isinstance(const, type) and issubclass(const, Exception)

    def _break_constant(self, interp, const):
        """
        Break down constant exception.
        """
        if isinstance(const, Exception):
            return const.__class__, const.args
        elif self._is_exception_type(const):
            return const, None
        else:
            raise NotImplementedError

    def _break_call(self, interp, call):
        """
        Break down call expression giving an exception instance.
        """
        if call.kws or call.vararg:
            raise NotImplementedError
        try:
            exc_type = interp.get_definition(call.func).infer_constant()
            exc_args = tuple(interp.get_definition(arg).infer_constant()
                             for arg in call.args)
        except (KeyError, TypeError):
            # Not all exception arguments are constants
            raise NotImplementedError
        return exc_type, exc_args

    def match(self, interp, block, typemap, calltypes):
        self.raises = raises = {}
        self.block = block
        # Detect all raise statements and find which ones can be
        # rewritten
        for inst in block.find_insts(ir.Raise):
            if inst.exception is None:
                # re-reraise
                exc_type, exc_args = None, None
            else:
                # raise <something> => find the definition site for <something>
                try:
                    excdef = interp.get_definition(inst.exception)
                except KeyError:
                    continue
                if isinstance(excdef, ir.Expr) and excdef.op == 'call':
                    # Is it the result of calling a constant?
                    try:
                        exc_type, exc_args = self._break_call(interp, excdef)
                    except NotImplementedError:
                        continue
                else:
                    # Is it a compile-time constant?
                    try:
                        const = excdef.infer_constant()
                    except TypeError:
                        continue
                    try:
                        exc_type, exc_args = self._break_constant(interp, const)
                    except NotImplementedError:
                        continue
            raises[inst] = exc_type, exc_args

        return len(raises) > 0

    def apply(self):
        """
        Rewrite all matching setitems as static_setitems.
        """
        new_block = self.block.copy()
        new_block.clear()
        for inst in self.block.body:
            if inst in self.raises:
                exc_type, exc_args = self.raises[inst]
                new_inst = ir.StaticRaise(exc_type, exc_args, inst.loc)
                new_block.append(new_inst)
            else:
                new_block.append(inst)
        return new_block
