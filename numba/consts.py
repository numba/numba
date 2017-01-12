from __future__ import print_function, absolute_import

from types import ModuleType

import weakref

from . import ir
from .errors import ConstantInferenceError


class ConstantInference(object):
    """
    A constant inference engine for a given interpreter.
    Inference inspects the IR to try and compute a compile-time constant for
    a variable.

    This shouldn't be used directly, instead call Interpreter.infer_constant().
    """

    def __init__(self, func_ir):
        # Avoid cyclic references as some user-visible objects may be
        # held alive in the cache
        self._func_ir = weakref.proxy(func_ir)
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
            defn = self._func_ir.get_definition(name)
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
        # Infer an expression: handle supported cases
        if expr.op == 'call':
            func = self.infer_constant(expr.func.name)
            return self._infer_call(func, expr)
        elif expr.op == 'getattr':
            value = self.infer_constant(expr.value.name)
            return self._infer_getattr(value, expr)
        elif expr.op == 'build_list':
            return [self.infer_constant(i.name) for i in expr.items]
        elif expr.op == 'build_tuple':
            return tuple(self.infer_constant(i.name) for i in expr.items)
        self._fail(expr)

    def _infer_call(self, func, expr):
        if expr.kws or expr.vararg:
            self._fail(expr)
        # Check supported callables
        if (func in (slice,) or
            (isinstance(func, type) and issubclass(func, BaseException))):
            args = [self.infer_constant(a.name) for a in expr.args]
            return func(*args)
        self._fail(expr)

    def _infer_getattr(self, value, expr):
        if isinstance(value, (ModuleType, type)):
            # Allow looking up a constant on a class or module
            return getattr(value, expr.attr)
        self._fail(expr)
