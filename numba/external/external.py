from numba.minivect import minitypes

class ExternalFunction(object):
    _attributes = ('func_name', 'arg_types', 'return_type', 'is_vararg')
    func_name = None
    arg_types = None
    return_type = None
    is_vararg = False

    badval = None
    goodval = None
    exc_type = None
    exc_msg = None
    exc_args = None

    def __init__(self, **kwargs):
        if __debug__:
            # Only accept keyword arguments defined _attributes
            for k, v in kwargs.items():
                if k not in self._attributes:
                    raise TypeError("Invalid keyword arg %s -> %s" % (k, v))
        vars(self).update(kwargs)

    @property
    def signature(self):
        return minitypes.FunctionType(return_type=self.return_type,
                                      args=self.arg_types,
                                      is_vararg=self.is_vararg)

    @property
    def name(self):
        if self.func_name is None:
            return type(self).__name__
        else:
            return self.func_name

class ExternalLibrary(object):
    def __init__(self, context):
        self._context = context
        # (name) -> (external function instance)
        self._functions = {}

    def add(self, extfn):
        if __debug__:
            # Sentry for duplicated external function name
            if extfn.name in self._functions:
                raise NameError("Duplicated external function: %s" % extfn.name)
        self._functions[extfn.name] = extfn

    def get(self, name):
        return self._functions[name]

    def __contains__(self, name):
        return name in self._functions

    def declare(self, module, name, arg_types=(), return_type=None):
        extfn = self._functions[name] # raises KeyError if name is not found

        if arg_types and return_type:
            if (extfn.arg_types != arg_types
                and extfn.return_type != return_type):
                raise TypeError("Signature mismatch")

        sig = extfn.signature
        lfunc_type = sig.to_llvm(self._context)
        return sig, module.get_or_insert_function(lfunc_type, extfn.name)



