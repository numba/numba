
import weakref

from . import ir
from .errors import ConstantInferenceError


class ConstantInference(object):
    """
    A constant inference object for a given interpreter.

    This shouldn't be used directly, instead call Interpreter.infer_constant().
    """

    def __init__(self, interp):
        # Avoid cyclic references as some user-visible objects may be
        # held alive in the cache
        self._interp = weakref.proxy(interp)
        self._cache = {}

    def infer_constant(self, name):
        """
        Infer a constant value for the given variable *name*.
        If no value can be inferred, numba.errors.ConstantInferenceError
        is raised.
        """
        if name not in self._cache:
            try:
                self._cache[name] = (True, self._do_infer(name))
            except ConstantInferenceError as exc:
                # Store the exception args only, to avoid keeping
                # a whole traceback alive.
                self._cache[name] = (False, (exc.__class__, exc.args))
        success, val = self._cache[name]
        if success:
            return val
        else:
            exc, args = val
            raise exc(*args)

    def _fail(self, val):
        raise ConstantInferenceError(
            "constant inference not possible for %s" % (val,))

    def _do_infer(self, name):
        if not isinstance(name, str):
            raise TypeError("infer_constant() called with non-str %r"
                            % (name,))
        try:
            defn = self._interp.get_definition(name)
        except KeyError:
            raise ConstantInferenceError(
                "no single definition for %r" % (name,))
        try:
            const = defn.infer_constant()
        except ConstantInferenceError:
            if isinstance(defn, ir.Expr):
                return self._infer_expr(defn)
            self._fail(defn)
        return const

    def _infer_expr(self, expr):
        if expr.op == 'call':
            if not expr.kws and not expr.vararg:
                func = self.infer_constant(expr.func.name)
                return self._infer_call(func, expr)
        self._fail(expr)

    def _infer_call(self, func, expr):
        if expr.kws or expr.vararg:
            self._fail(expr)
        if (func in (slice,) or
            (isinstance(func, type) and issubclass(func, BaseException))):
            args = [self.infer_constant(a.name) for a in expr.args]
            return func(*args)
        self._fail(expr)
