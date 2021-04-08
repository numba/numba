try:
    # This is usually the the first C extension import performed when importing
    # Numba, if it fails to import, provide some feedback
    from numba.core.typeconv import _typeconv
except ImportError as e:
    from numba.core.errors import feedback_details as reportme
    import sys
    url = "https://numba.pydata.org/numba-doc/latest/developer/contributing.html"
    dashes = '-' * 80
    msg = ("Numba could not be imported.\nIf you are seeing this message and "
           "are undertaking Numba development work, you may need to re-run:\n\n"
           "python setup.py build_ext --inplace\n\n(Also, please check the "
           "development set up guide %s.)\n\nIf you are not working on Numba "
           "development:\n%s\nThe original error was: '%s'\n" + dashes + "\nIf "
           "possible please include the following in your error report:\n\n"
           "sys.executable: %s\n")
    raise ImportError(msg % (url, reportme, str(e), sys.executable))

from numba.core.typeconv import castgraph, Conversion
from numba.core import types


class TypeManager(object):

    # The character codes used by the C/C++ API (_typeconv.cpp)
    _conversion_codes = {
        Conversion.safe: ord("s"),
        Conversion.unsafe: ord("u"),
        Conversion.promote: ord("p"),
        }

    def __init__(self):
        self._ptr = _typeconv.new_type_manager()
        self._types = set()

    def select_overload(self, sig, overloads, allow_unsafe,
                        exact_match_required):
        sig = [t._code for t in sig]
        overloads = [[t._code for t in s] for s in overloads]
        return _typeconv.select_overload(self._ptr, sig, overloads,
                                         allow_unsafe, exact_match_required)

    def check_compatible(self, fromty, toty):
        if not isinstance(toty, types.Type):
            raise ValueError("Specified type '%s' (%s) is not a Numba type" %
                             (toty, type(toty)))
        name = _typeconv.check_compatible(self._ptr, fromty._code, toty._code)
        conv = Conversion[name] if name is not None else None
        assert conv is not Conversion.nil
        return conv

    def set_compatible(self, fromty, toty, by):
        code = self._conversion_codes[by]
        _typeconv.set_compatible(self._ptr, fromty._code, toty._code, code)
        # Ensure the types don't die, otherwise they may be recreated with
        # other type codes and pollute the hash table.
        self._types.add(fromty)
        self._types.add(toty)

    def set_promote(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.promote)

    def set_unsafe_convert(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.unsafe)

    def set_safe_convert(self, fromty, toty):
        self.set_compatible(fromty, toty, Conversion.safe)

    def get_pointer(self):
        return _typeconv.get_pointer(self._ptr)


class TypeCastingRules(object):
    """
    A helper for establishing type casting rules.
    """
    def __init__(self, tm):
        self._tm = tm
        self._tg = castgraph.TypeGraph(self._cb_update)

    def promote(self, a, b):
        """
        Set `a` can promote to `b`
        """
        self._tg.promote(a, b)

    def unsafe(self, a, b):
        """
        Set `a` can unsafe convert to `b`
        """
        self._tg.unsafe(a, b)

    def safe(self, a, b):
        """
        Set `a` can safe convert to `b`
        """
        self._tg.safe(a, b)

    def promote_unsafe(self, a, b):
        """
        Set `a` can promote to `b` and `b` can unsafe convert to `a`
        """
        self.promote(a, b)
        self.unsafe(b, a)

    def safe_unsafe(self, a, b):
        """
        Set `a` can safe convert to `b` and `b` can unsafe convert to `a`
        """
        self._tg.safe(a, b)
        self._tg.unsafe(b, a)

    def unsafe_unsafe(self, a, b):
        """
        Set `a` can unsafe convert to `b` and `b` can unsafe convert to `a`
        """
        self._tg.unsafe(a, b)
        self._tg.unsafe(b, a)

    def _cb_update(self, a, b, rel):
        """
        Callback for updating.
        """
        if rel == Conversion.promote:
            self._tm.set_promote(a, b)
        elif rel == Conversion.safe:
            self._tm.set_safe_convert(a, b)
        elif rel == Conversion.unsafe:
            self._tm.set_unsafe_convert(a, b)
        else:
            raise AssertionError(rel)

