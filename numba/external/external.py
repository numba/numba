from numba.minivect import minitypes

class ExternalFunction(object):
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
        self._functions[extfn.name] = extfn

    def declare(self, module, name, arg_types=(), return_type=None):
        extfn = self._functions[name] # raises KeyError if name is not found

        if arg_types and return_type:
            if (extfn.arg_types != arg_types
                and extfn.return_type != return_type):
                raise TypeError("Signature mismatch")

        sig = extfn.signature
        lfunc_type = sig.to_llvm(self._context)
        return sig, module.get_or_insert_function(lfunc_type, extfn.name)



