#! /usr/bin/env python
# ______________________________________________________________________

import sys
import os.path
import imp
import io
import types

import llvm.core as lc
import llvm.ee as le

import bytetype
import byte_translator
from pyaddfunc import pyaddfunc

LLVM_TO_INT_PARSE_STR_MAP = {
    8 : 'b',
    16 : 'h', 
    32 : 'i', # Note that on 32-bit systems sizeof(int) == sizeof(long)
    64 : 'L', # Seeing sizeof(long long) == 8 on both 32 and 64-bit platforms
}

LLVM_TO_PARSE_STR_MAP = {
    lc.TYPE_FLOAT : 'f',
    lc.TYPE_DOUBLE : 'd',
}

# ______________________________________________________________________

# XXX Stolen from numba.translate

def get_string_constant (module, const_str):
    const_name = "__STR_%x" % (hash(const_str),)
    try:
        ret_val = module.get_global_variable_named(const_name)
    except:
        lconst_str = lc.Constant.stringz(const_str)
        ret_val = module.add_global_variable(lconst_str.type, const_name)
        ret_val.initializer = lconst_str
        ret_val.linkage = lc.LINKAGE_INTERNAL
    return ret_val

# ______________________________________________________________________

class NoBitey (object):
    def __init__ (self, target_module = None, type_annotations = None):
        if target_module is None:
            target_module = lc.Module.new('NoBitey_%d' % id(self))
        if type_annotations is None:
            type_annotations = {}
        self.target_module = target_module
        self.type_aliases = type_annotations # Reserved for future use.

    def _build_parse_string (self, llvm_type):
        kind = llvm_type.kind
        if kind == lc.TYPE_INTEGER:
            ret_val = LLVM_TO_INT_PARSE_STR_MAP[llvm_type.width]
        elif kind in LLVM_TO_PARSE_STR_MAP:
            ret_val = LLVM_TO_PARSE_STR_MAP[kind]
        else:
            raise TypeError('Unsupported LLVM type: %s' % str(llvm_type))
        return ret_val

    def build_parse_string (self, llvm_tys):
        """Given a set of LLVM types, return a string for parsing
        them via PyArg_ParseTuple."""
        return ''.join((self._build_parse_string(ty)
                        for ty in llvm_tys))

    def handle_abi_casts (self, builder, result):
        if result.type.kind == lc.TYPE_FLOAT:
            # NOTE: The C ABI apparently casts floats to doubles when
            # an argument must be pushed on the stack, as is the case
            # when calling a variable argument function.
            # XXX Is there documentation on this where I can find all
            # coercion rules?  Do we still need some libffi
            # integration?
            result = builder.fpext(result, bytetype.ldouble)
        return result

    def build_wrapper_function (self, llvm_function, engine = None):
        arg_types = llvm_function.type.pointee.args
        return_type = llvm_function.type.pointee.return_type
        li32_0 = lc.Constant.int(bytetype.li32, 0)
        def get_llvm_function (builder):
            if self.target_module != llvm_function.module:
                llvm_function_ptr = self.target_module.add_global_variable(
                    llvm_function.type, llvm_function.name)
                llvm_function_ptr.initializer = lc.Constant.inttoptr(
                    lc.Constant.int(
                        bytetype.liptr,
                        engine.get_pointer_to_function(llvm_function)),
                    llvm_function.type)
                llvm_function_ptr.linkage = lc.LINKAGE_INTERNAL
                ret_val = builder.load(llvm_function_ptr)
            else:
                ret_val = llvm_function
            return ret_val
        def build_parse_args (builder):
            return [builder.alloca(arg_type) for arg_type in arg_types]
        def build_parse_string (builder):
            parse_str = get_string_constant(
                self.target_module, self.build_parse_string(arg_types))
            return builder.gep(parse_str, (li32_0, li32_0))
        def load_target_args (builder, args):
            return [builder.load(arg) for arg in args]
        def build_build_string (builder):
            build_str = get_string_constant(
                self.target_module, self._build_parse_string(return_type))
            return builder.gep(build_str, (li32_0, li32_0))
        handle_abi_casts = self.handle_abi_casts
        target_function_name = llvm_function.name + "_wrapper"
        # __________________________________________________
        @byte_translator.llnumba(bytetype.l_pyfunc, self.target_module,
                                 **locals())
        def _wrapper (self, args):
            ret_val = l_pyobj_p(0)
            parse_args = build_parse_args()
            parse_result = PyArg_ParseTuple(args, build_parse_string(),
                                            *parse_args)
            if parse_result != li32(0):
                thread_state = PyEval_SaveThread()
                target_args = load_target_args(parse_args)
                llresult = handle_abi_casts(get_llvm_function()(*target_args))
                PyEval_RestoreThread(thread_state)
                ret_val = Py_BuildValue(build_build_string(), llresult)
            return ret_val
        # __________________________________________________
        return _wrapper

    def wrap_llvm_module (self, llvm_module, engine = None, py_module = None):
        '''
        Shamefully adapted from bitey.bind.wrap_llvm_module().
        '''
        functions = [func for func in llvm_module.functions
                     if not func.name.startswith("_")
                     and not func.is_declaration
                     and func.linkage == lc.LINKAGE_EXTERNAL]
        if engine is None:
            engine = le.ExecutionEngine.new(llvm_module)
        wrappers = [self.build_wrapper_function(func, engine)
                    for func in functions]
        if __debug__: print(self.target_module)
        if self.target_module != llvm_module:
            engine.add_module(self.target_module)
        py_wrappers = [pyaddfunc(wrapper.name,
                                 engine.get_pointer_to_function(wrapper))
                       for wrapper in wrappers]
        if py_module:
            for py_wrapper in py_wrappers:
                setattr(py_module, py_wrapper.__name__[:-8], py_wrapper)
            setattr(py_module, '_llvm_module', llvm_module)
            setattr(py_module, '_llvm_engine', engine)
            if self.target_module != llvm_module:
                setattr(py_module, '_llvm_wrappers', self.target_module)
        return engine, py_wrappers

    def wrap_llvm_module_in_python (self, llvm_module, py_module = None):
        '''
        Mildly reworked and abstracted bitey.bind.wrap_llvm_bitcode().
        Abstracted to accept any existing LLVM Module object, and
        return a Python wrapper module (even if one wasn't originally
        specified).
        '''
        if py_module is None:
            py_module = types.ModuleType(str(llvm_module.id))
        engine = le.ExecutionEngine.new(llvm_module)
        self.wrap_llvm_module(llvm_module, engine, py_module)
        return py_module

    def wrap_llvm_bitcode (self, bitcode, py_module = None):
        '''
        Intended to be drop-in replacement of
        bitey.bind.wrap_llvm_bitcode().
        '''
        return self.wrap_llvm_module_in_python(
            lc.Module.from_bitcode(io.BytesIO(bitcode)), py_module)

    def wrap_llvm_assembly (self, llvm_asm, py_module = None):
        return self.wrap_llvm_module_in_python(
            lc.Module.from_assembly(io.BytesIO(llvm_asm)), py_module)

# ______________________________________________________________________

class NoBiteyLoader(object):
    """
    Load LLVM compiled bitcode and autogenerate a ctypes binding.

    Initially copied and adapted from bitey.loader module.
    """
    def __init__(self, pkg, name, source, preload, postload):
        self.package = pkg
        self.name = name
        self.fullname = '.'.join((pkg,name))
        self.source = source
        self.preload = preload
        self.postload = postload

    @classmethod
    def _check_magic(cls, filename):
        if os.path.exists(filename):
            magic = open(filename,"rb").read(4)
            if magic == b'\xde\xc0\x17\x0b':
                return True
            elif magic[:2] == b'\x42\x43':
                return True
            else:
                return False
        else:
            return False

    @classmethod
    def build_module(cls, fullname, source_path, source_data, preload=None,
                     postload=None):
        name = fullname.split(".")[-1]
        mod = imp.new_module(name)
        if preload:
            exec(preload, mod.__dict__, mod.__dict__)
        type_annotations = getattr(mod, '_type_annotations', None)
        nb = NoBitey(type_annotations = type_annotations)
        if source_path.endswith(('.o', '.bc')):
            nb.wrap_llvm_bitcode(source_data, mod)
        elif source_path.endswith('.s'):
            nb.wrap_llvm_assembly(source_data, mod)
        if postload:
            exec(postload, mod.__dict__, mod.__dict__)
        return mod

    @classmethod
    def find_module(cls, fullname, paths = None):
        if paths is None:
            paths = sys.path
        names = fullname.split('.')
        modname = names[-1]
        source_paths = None
        for f in paths:
            path = os.path.join(os.path.realpath(f), modname)
            source = path + '.o'
            if cls._check_magic(source):
                source_paths = path, source
                break
            source = path + '.bc'
            if os.path.exists(source):
                source_paths = path, source
                break
            source = path + '.s'
            if os.path.exists(source):
                source_paths = path, source
                break
        if source_paths:
            path, source = source_paths
            return cls('.'.join(names[:-1]), modname, source,
                       path + ".pre.py", path + ".post.py")

    def get_code(self, module):
        pass

    def get_data(self, module):
        pass

    def get_filename(self, name):
        return self.source

    def get_source(self, name):
        with open(self.source, 'rb') as f:
             return f.read()

    def is_package(self, *args, **kw):
        return False

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        preload = None
        postload = None

        # Get the preload file (if any)
        if os.path.exists(self.preload):
            with open(self.preload) as f:
                preload = f.read()

        # Get the source
        with open(self.source, 'rb') as f:
            source_data = f.read()

        # Get the postload file (if any)
        if os.path.exists(self.postload):
            with open(self.postload) as f:
                postload = f.read()

        mod = self.build_module(fullname, self.get_filename(None), source_data,
                                preload, postload)
        sys.modules[fullname] = mod
        mod.__loader__ = self
        mod.__file__ = self.source
        return mod

    @classmethod
    def install(cls):
        if cls not in sys.meta_path:
            sys.meta_path.append(cls)

    @classmethod
    def remove(cls):
        sys.meta_path.remove(cls)

# ______________________________________________________________________

def _mk_add_42 (llvm_module, at_type = bytetype.lc_long):
    f = llvm_module.add_function(
        lc.Type.function(at_type, [at_type]), 'add_42_%s' % str(at_type))
    block = f.append_basic_block('entry')
    builder = lc.Builder.new(block)
    if at_type.kind == lc.TYPE_INTEGER:
        const_42 = lc.Constant.int(at_type, 42)
        add = builder.add
    elif at_type.kind in (lc.TYPE_FLOAT, lc.TYPE_DOUBLE):
        const_42 = lc.Constant.real(at_type, 42.)
        add = builder.fadd
    else:
        raise TypeError('Unsupported type: %s' % str(at_type))
    builder.ret(add(f.args[0], const_42))
    return f

# ______________________________________________________________________

def build_test_module ():
    llvm_module = lc.Module.new('nobitey_test')
    for ty in (bytetype.li32, bytetype.li64, bytetype.lfloat,
               bytetype.ldouble):
        fn = _mk_add_42(llvm_module, ty)
    return llvm_module

# ______________________________________________________________________

def test_wrap_module (arg = None):
    # Build up a module.
    m = build_test_module()
    if arg and arg.lower() == 'separated':
        wrap_module = NoBitey().wrap_llvm_module_in_python(m)
    else:
        wrap_module = NoBitey(m).wrap_llvm_module_in_python(m)
    # Now try running the generated wrappers.
    for py_wf_name in ('add_42_i32', 'add_42_i64', 'add_42_float',
                       'add_42_double'):
        py_wf = getattr(wrap_module, py_wf_name)
        for i in range(42):
            result = py_wf(i)
            expected = i + 42
            assert result == expected, "%r != %r in %r" % (
                result, expected, py_wf)
    return wrap_module

# ______________________________________________________________________

def main (*args):
    if args:
        for arg in args:
            test_wrap_module(arg)
    else:
        test_wrap_module()

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of nobitey.py
