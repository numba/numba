from . import fnlib, imlib, types

def extends(libs, exts):
    funclib, implib = libs
    for extcls in exts:
        ext = Extension(extcls())
        ext.declare(funclib)
        ext.define(implib)
    return funclib, implib

neither_function_or_method = Exception('neither function or method')

class Extension(object):
    def __init__(self, ext):
        self.ext = ext
        self.is_method = hasattr(self.ext, 'method')
        self.is_function = hasattr(self.ext, 'function')
        assert self.is_method or self.is_function
        assert not (self.is_method and self.is_function)
        self.signature = (self.ext.method
                          if self.is_method
                          else self.ext.function)

    @property
    def funcobj(self):
        return self.signature[0]

    @property
    def args(self):
        return self.signature[1]

    @property
    def return_type(self):
        return self.signature[2]

    def declare(self, lib):
        if self.is_function:
            lib.define(fnlib.Function(funcobj=self.funcobj,
                                      args=self.args,
                                      return_type=self.return_type))
        elif self.is_method:
            lib.define(fnlib.Function(funcobj='.%s' % self.funcobj,
                                      args=self.args,
                                      return_type=types.method_type))
            lib.define(fnlib.Function(funcobj='@%s' % self.funcobj,
                                      args=self.args,
                                      return_type=self.return_type))
        else:
            raise neither_function_or_method

    @property
    def implementor(self):
        fnorder = 'generic_implement', 'implement'
        for fname in fnorder:
            fn = getattr(self.ext, fname, None)
            return fn
        raise Exception('extension %s does not define an implementor' %
                        self.ext)

    def define(self, lib):
        if self.is_function:
            funcobj = self.funcobj
        elif self.is_method:
            funcobj = '@%s' % self.funcobj
        else:
            raise neither_function_or_method
        lib.define(imlib.Imp(self.implementor, funcobj, args=self.args))

