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

    def build_wrapper_function (self, llvm_function):
        _pyobj_p = bytetype.l_pyobject_head_struct_p
        _void_p = _char_p = bytetype.li8_ptr
        self.crnt_function = self.target_module.add_function(
            lc.Type.function(_pyobj_p, (_pyobj_p, _pyobj_p)),
            llvm_function.name + "_wrapper")
        entry_block = self.crnt_function.append_basic_block('entry')
        args_ok_block = self.crnt_function.append_basic_block('args_ok')
        exit_block = self.crnt_function.append_basic_block('exit')
        _int32_zero = lc.Constant.int(bytetype.li32, 0)
        _Py_BuildValue = self.target_module.get_or_insert_function(
            lc.Type.function(_pyobj_p, [_char_p], True), 'Py_BuildValue')
        _PyArg_ParseTuple = self.target_module.get_or_insert_function(
            lc.Type.function(bytetype.lc_int, [_pyobj_p, _char_p], True),
            'PyArg_ParseTuple')
        _PyEval_SaveThread = self.target_module.get_or_insert_function(
            lc.Type.function(_void_p, []), 'PyEval_SaveThread')
        _PyEval_RestoreThread = self.target_module.get_or_insert_function(
            lc.Type.function(lc.Type.void(), [_void_p]),
            'PyEval_RestoreThread')
        # __________________________________________________
        # entry:
        builder = lc.Builder.new(entry_block)
        arg_types = llvm_function.type.pointee.args
        parse_str = builder.gep(
            get_string_constant(
                self.target_module,
                self.build_parse_string(arg_types)),
            [_int32_zero, _int32_zero])
        parse_args = [builder.alloca(arg_ty) for arg_ty in arg_types]
        parse_args.insert(0, parse_str)
        parse_args.insert(0, self.crnt_function.args[1])
        parse_result = builder.call(_PyArg_ParseTuple, parse_args)
        builder.cbranch(builder.icmp(lc.ICMP_NE, parse_result, _int32_zero),
                        args_ok_block, exit_block)
        # __________________________________________________
        # args_ok:
        builder = lc.Builder.new(args_ok_block)
        thread_state = builder.call(_PyEval_SaveThread, ())
        target_args = [builder.load(parse_arg) for parse_arg in parse_args[2:]]
        result = builder.call(llvm_function, target_args)
        result_cast = self.handle_abi_casts(builder, result)
        builder.call(_PyEval_RestoreThread, (thread_state,))
        build_str = builder.gep(
            get_string_constant(
                self.target_module,
                self._build_parse_string(result.type)),
            [_int32_zero, _int32_zero])
        py_result = builder.call(_Py_BuildValue, [build_str, result_cast])
        builder.branch(exit_block)
        # __________________________________________________
        # exit:
        builder = lc.Builder.new(exit_block)
        rval = builder.phi(bytetype.l_pyobject_head_struct_p)
        rval.add_incoming(lc.Constant.null(bytetype.l_pyobject_head_struct_p),
                          entry_block)
        rval.add_incoming(py_result, args_ok_block)
        builder.ret(rval)
        return self.crnt_function

    def wrap_llvm_module (self, llvm_module, engine = None, py_module = None):
        '''
        Shamefully adapted from bitey.bind.wrap_llvm_module().
        '''
        functions = [func for func in llvm_module.functions
                     if not func.name.startswith("_")
                     and not func.is_declaration
                     and func.linkage == lc.LINKAGE_EXTERNAL]
        wrappers = [self.build_wrapper_function(func) for func in functions]
        if engine is None:
            engine = le.ExecutionEngine.new(llvm_module)
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

def main (*args):
    # Build up a module.
    m = build_test_module()
    print(m)
    wrap_module = NoBitey().wrap_llvm_module_in_python(m)
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

if __name__ == "__main__":
    main(*sys.argv[1:])

# ______________________________________________________________________
# End of nobitey.py
