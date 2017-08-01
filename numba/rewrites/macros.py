from numba import ir, errors
from . import register_rewrite, Rewrite


class Macro(object):
    '''
    A macro object is expanded to a function call

    Args
    ----
    name: str
        Name of this Macro
    func: function
        Function that evaluates the macro expansion.
    callable: bool
        True if the macro is callable from Python code
        (``func`` is then a Python callable returning the desired IR node).
        False if the macro is not callable
        (``func`` is then the name of a backend-specific function name
         specifying the function to call at runtime).
    argnames: list
        If ``callable`` is True, this holds a list of the names of arguments
        to the function.
    '''

    __slots__ = 'name', 'func', 'callable', 'argnames'

    def __init__(self, name, func, callable=False, argnames=None):
        self.name = name
        self.func = func
        self.callable = callable
        self.argnames = argnames

    def __repr__(self):
        return '<macro %s -> %s>' % (self.name, self.func)


@register_rewrite('before-inference')
class ExpandMacros(Rewrite):
    """
    Expand lookups and calls of Macro objects.
    """

    def match(self, func_ir, block, typemap, calltypes):
        """
        Look for potential macros for expand and store their expansions.
        """
        self.block = block
        self.rewrites = rewrites = {}

        for inst in block.body:
            if isinstance(inst, ir.Assign):
                rhs = inst.value
                if (isinstance(rhs, ir.Expr) and rhs.op == 'call'
                    and isinstance(rhs.func, ir.Var)):
                    # Is it a callable macro?
                    try:
                        const = func_ir.infer_constant(rhs.func)
                    except errors.ConstantInferenceError:
                        continue
                    if isinstance(const, Macro):
                        assert const.callable
                        new_expr = self._expand_callable_macro(func_ir, rhs,
                                                               const, rhs.loc)
                        rewrites[rhs] = new_expr

                elif isinstance(rhs, ir.Expr) and rhs.op == 'getattr':
                    # Is it a non-callable macro looked up as a constant attribute?
                    try:
                        const = func_ir.infer_constant(inst.target)
                    except errors.ConstantInferenceError:
                        continue
                    if isinstance(const, Macro) and not const.callable:
                        new_expr = self._expand_non_callable_macro(const, rhs.loc)
                        rewrites[rhs] = new_expr

        return len(rewrites) > 0

    def _expand_non_callable_macro(self, macro, loc):
        """
        Return the IR expression of expanding the non-callable macro.
        """
        intr = ir.Intrinsic(macro.name, macro.func, args=())
        new_expr = ir.Expr.call(func=intr, args=(),
                                kws=(), loc=loc)
        return new_expr

    def _expand_callable_macro(self, func_ir, call, macro, loc):
        """
        Return the IR expression of expanding the macro call.
        """
        assert macro.callable

        # Resolve all macro arguments as constants, or fail
        args = [func_ir.infer_constant(arg.name) for arg in call.args]
        kws = {}
        for k, v in call.kws:
            try:
                kws[k] = func_ir.infer_constant(v)
            except errors.ConstantInferenceError:
                msg = "Argument {name!r} must be a " \
                      "constant at {loc}".format(name=k,
                                                 loc=loc)
                raise ValueError(msg)

        try:
            result = macro.func(*args, **kws)
        except Exception as e:
            msg = str(e)
            headfmt = "Macro expansion failed at {line}"
            head = headfmt.format(line=loc)
            newmsg = "{0}:\n{1}".format(head, msg)
            raise errors.MacroError(newmsg)

        assert result is not None

        result.loc = call.loc
        new_expr = ir.Expr.call(func=result, args=call.args,
                                kws=call.kws, loc=loc)
        return new_expr

    def apply(self):
        """
        Apply the expansions computed in .match().
        """
        block = self.block
        rewrites = self.rewrites
        for inst in block.body:
            if isinstance(inst, ir.Assign) and inst.value in rewrites:
                inst.value = rewrites[inst.value]
        return block
