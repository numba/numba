import logging
import os
import sys
import functools
from importlib import import_module
from numba import decorators
from numba import llvm_types
import llvm.core as lc
logger = logging.getLogger(__name__)

__all__ = ['which', 'find_linker', 'find_args', 'find_shared_ending',
           'Compiler',
           ]

NULL = lc.Constant.null(llvm_types._void_star)
zero = lc.Constant.int(llvm_types._int32, 0)
METH_VARARGS = lc.Constant.int(llvm_types._int32, 1) # == METH_VARARGS
method_def_ty = lc.Type.struct(
    (llvm_types._int8_star, llvm_types._void_star, llvm_types._int32,
     llvm_types._int8_star))
method_def_ptr = lc.Type.pointer(method_def_ty)
py_init_module_fn_ty = lc.Type.function(llvm_types._pyobject_head_struct_p, [
        llvm_types._int8_star, method_def_ptr, llvm_types._int8_star,
        llvm_types._pyobject_head_struct_p, llvm_types._int32])

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, fname)
            if is_exe(exe_file):
                return exe_file
    return None

_configs = {'win' : ("link.exe", ("/dll",), '.dll'),
    'dar': ("libtool", ("-dynamic",), '.dylib'),
    'default': ("ld", ("-shared",), ".so")
}

def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]

find_linker = functools.partial(get_configs, 0)
find_args = functools.partial(get_configs, 1)
find_shared_ending = functools.partial(get_configs, 2)

class Compiler(object):
    def __init__(self, inputs, module_name='numba_exported'):
        self.inputs = inputs
        self.exported_signatures = {}
        self.module_name = module_name

    def _emit_wrapper_init(self, llvm_module, method_defs):
        # Figure out the Python C API module creation function, and
        # get a LLVM function for it.
        py_init_module_fn_name = "Py_InitModule4"
        if llvm_types._trace_refs_:
            py_init_module_fn_name += "TraceRefs"
        if llvm_types._plat_bits == 64:
            py_init_module_fn_name += "_64"
        py_init_module_fn = llvm_module.add_function(py_init_module_fn_ty,
                                                     py_init_module_fn_name)
        py_init_module_fn.linkage = lc.LINKAGE_EXTERNAL
        # Define a constant string for the module name.
        module_name_init = lc.Constant.stringz(self.module_name)
        module_name_const = llvm_module.add_global_variable(
            module_name_init.type, '.module_name')
        module_name_const.initializer = module_name_init
        module_name_const.linkage = lc.LINKAGE_INTERNAL
        # Finish off the method definition array, and construct it as
        # a module constant.
        sentinel = lc.Constant.struct([NULL, NULL, zero, NULL])
        method_defs.append(sentinel)
        method_array_init = lc.Constant.array(method_def_ty, method_defs)
        method_array = llvm_module.add_global_variable(method_array_init.type,
                                                       '.module_methods')
        method_array.initializer = method_array_init
        method_array.linkage = lc.LINKAGE_INTERNAL
        method_array_ptr = lc.Constant.gep(method_array, [zero, zero])
        # Define the module initialization function.
        module_init_fn_name = "init" + self.module_name
        module_init_fn_ty = lc.Type.function(lc.Type.void(), ())
        module_init_fn = llvm_module.add_function(module_init_fn_ty,
                                                  module_init_fn_name)
        entry = module_init_fn.append_basic_block('Entry')
        builder = lc.Builder.new(entry)
        builder.call(py_init_module_fn,
                     (lc.Constant.gep(module_name_const, [zero, zero]),
                      method_array_ptr,
                      NULL,
                      lc.Constant.null(llvm_types._pyobject_head_struct_p),
                      lc.Constant.int(llvm_types._int32, sys.api_version)))
        builder.ret_void()

    def _cull_exports(self):
        '''
        Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.

        Resets the export environment.
        '''
        exports_env = decorators.pipeline_env.exports
        self.exported_signatures = exports_env.function_signature_map
        ret_val = lc.Module.new(self.module_name)
        for submod in exports_env.function_module_map.values():
            ret_val.link_in(submod)
        if exports_env.wrap:
            method_defs = []
            wrappers = exports_env.function_wrapper_map.items()
            for name, (submod, lfunc) in wrappers:
                ret_val.link_in(submod)
                method_name_init = lc.Constant.stringz(name)
                method_name = ret_val.add_global_variable(
                    method_name_init.type, '.method_name')
                method_name.initializer = method_name_init
                method_name.linkage = lc.LINKAGE_INTERNAL
                method_def_const = lc.Constant.struct([
                        lc.Constant.gep(method_name, [zero, zero]),
                        lc.Constant.bitcast(lfunc, llvm_types._void_star),
                        METH_VARARGS, NULL])
                method_defs.append(method_def_const)
            self._emit_wrapper_init(ret_val, method_defs)
        decorators._init_exports()
        return ret_val

    def _process_inputs(self, wrap=False, **kws):
        decorators.pipeline_env.exports.wrap = wrap
        for ifile in self.inputs:
            execfile(ifile)

    def write_llvm_bitcode(self, output, **kws):
        self._process_inputs(**kws)
        lmod = self._cull_exports()
        with open(output, 'wb') as fout:
            lmod.to_bitcode(fout)

    def write_native_object(self, output, **kws):
        self._process_inputs(**kws)
        lmod = self._cull_exports()
        with open(output, 'wb') as fout:
            fout.write(lmod.to_native_object())

    def emit_header(self, output):
        from numba.minivect import minitypes

        fname, ext = os.path.splitext(output)
        with open(fname + '.h', 'wb') as fout:
            fout.write(minitypes.get_utility())
            fout.write("\n/* Prototypes */\n")
            for signature in self.exported_signatures.values():
                name = signature.name
                restype = signature.return_type.declare()
                args = ", ".join(argtype.declare()
                                 for argtype in signature.args)
                fout.write("extern %s %s(%s);\n" % (restype, name, args))
