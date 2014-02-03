# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import logging
import os
import sys
import functools

import llvm.core as lc
import llvm.ee as le
import llvm.passes as lp

from numba.utils import IS_PY3
from . import llvm_types as lt
from .decorators import registry as export_registry
from numba.compiler import compile_extra, Flags
from numba.targets.registry import CPUTarget

logger = logging.getLogger(__name__)

__all__ = ['which', 'find_linker', 'find_args', 'find_shared_ending', 'Compiler']

NULL = lc.Constant.null(lt._void_star)
ZERO = lc.Constant.int(lt._int32, 0)
ONE = lc.Constant.int(lt._int32, 1)
METH_VARARGS_AND_KEYWORDS = lc.Constant.int(lt._int32, 1|2)

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

_configs = {
    'win': ("link.exe", ("/dll",), '.dll'),
    'dar': ("libtool", ("-dynamic", "-undefined", "dynamic_lookup"), '.so'),
    'default': ("ld", ("-shared",), ".so")
}


def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]


find_linker = functools.partial(get_configs, 0)
find_args = functools.partial(get_configs, 1)
find_shared_ending = functools.partial(get_configs, 2)


class _Compiler(object):
    """A base class to compile Python modules to a single shared library or
    extension module.

    :param inputs: input file(s).
    :type inputs: iterable
    :param module_name: the name of the exported module.
    """

    #: Structure used to describe a method of an extension type.
    #: struct PyMethodDef {
    #:     const char  *ml_name;       /* The name of the built-in function/method */
    #:     PyCFunction  ml_meth;       /* The C function that implements it */
    #:     int          ml_flags;      /* Combination of METH_xxx flags, which mostly
    #:                                    describe the args expected by the C func */
    #:     const char  *ml_doc;        /* The __doc__ attribute, or NULL */
    #: };
    method_def_ty = lc.Type.struct((lt._int8_star,
                                    lt._void_star,
                                    lt._int32,
                                    lt._int8_star))

    method_def_ptr = lc.Type.pointer(method_def_ty)

    def __init__(self, inputs, module_name='numba_exported'):
        self.inputs = inputs
        self.module_name = module_name
        self.export_python_wrap = False

    def _emit_python_wrapper(self, llvm_module):
        """Emit generated Python wrapper and extension module code.
        """
        raise NotImplementedError

    def _cull_exports(self):
        """Read all the exported functions/modules in the translator
        environment, and join them into a single LLVM module.

        Resets the export environment afterwards.
        """
        self.exported_signatures = export_registry

        # Create new module containing everything
        llvm_module = lc.Module.new(self.module_name)

        # Compile all exported functions
        typing_ctx = CPUTarget.typing_context
        # TODO Use non JIT-ing target
        target_ctx = CPUTarget.target_context
        modules = []
        flags = Flags()
        if not self.export_python_wrap:
            flags.set("no_compile")

        for entry in self.exported_signatures:
            cres = compile_extra(typing_ctx, target_ctx, entry.function,
                                 entry.signature.args,
                                 entry.signature.return_type, flags,
                                 locals={})

            if self.export_python_wrap:
                module = cres.llvm_func.module
                cres.llvm_func.linkage = lc.LINKAGE_INTERNAL
                wrappername = "wrapper." + cres.llvm_func.name
                wrapper = module.get_function_named(wrappername)
                wrapper.name = entry.symbol
            else:
                cres.llvm_func.name = entry.symbol

            modules.append(cres.llvm_module)

        # Link all exported functions
        for mod in modules:
            llvm_module.link_in(mod, preserve=self.export_python_wrap)

        # Optimize
        tm = le.TargetMachine.new(opt=3)
        pms = lp.build_pass_managers(tm=tm, opt=3, loop_vectorize=True,
                                     fpm=False)
        pms.pm.run(llvm_module)

        if self.export_python_wrap:
            self._emit_python_wrapper(llvm_module)

        del self.exported_signatures[:]
        print(llvm_module)
        return llvm_module

    def _process_inputs(self, wrap=False, **kws):
        for ifile in self.inputs:
            with open(ifile) as fin:
                exec(compile(fin.read(), ifile, 'exec'))

        self.export_python_wrap = wrap

    def write_llvm_bitcode(self, output, **kws):
        self._process_inputs(**kws)
        lmod = self._cull_exports()
        with open(output, 'wb') as fout:
            lmod.to_bitcode(fout)

    def write_native_object(self, output, **kws):
        self._process_inputs(**kws)
        lmod = self._cull_exports()
        tm = le.TargetMachine.new(opt=3, reloc=le.RELOC_PIC, features='-avx')

        with open(output, 'wb') as fout:
            objfile = tm.emit_object(lmod)
            fout.write(objfile)

    def emit_type(self, tyobj):
        ret_val = str(tyobj)
        if hasattr(tyobj, 'kind') and tyobj.kind == 'int':
            if ret_val.endswith(('8', '16', '32', '64')):
                ret_val += "_t"
        return ret_val

    def emit_header(self, output):
        from numba.minivect import minitypes

        fname, ext = os.path.splitext(output)
        with open(fname + '.h', 'w') as fout:
            fout.write(minitypes.get_utility())
            fout.write("\n/* Prototypes */\n")
            for signature in self.exported_signatures.values():
                name = signature.name
                restype = self.emit_type(signature.return_type)
                args = ", ".join(self.emit_type(argtype)
                                 for argtype in signature.args)
                fout.write("extern %s %s(%s);\n" % (restype, name, args))

    def _emit_method_array(self, llvm_module):
        """Collect exported methods and emit a PyMethodDef array.

        :returns: a pointer to the PyMethodDef array.
        """
        method_defs = []
        for entry in self.exported_signatures:
            name = entry.symbol
            lfunc = llvm_module.get_function_named(name)

            method_name_init = lc.Constant.stringz(name)
            method_name = llvm_module.add_global_variable(
                method_name_init.type, '.method_name')
            method_name.initializer = method_name_init
            method_name.linkage = lc.LINKAGE_EXTERNAL
            method_def_const = lc.Constant.struct((lc.Constant.gep(method_name, [ZERO, ZERO]),
                                                   lc.Constant.bitcast(lfunc, lt._void_star),
                                                   METH_VARARGS_AND_KEYWORDS,
                                                   NULL))
            method_defs.append(method_def_const)

        sentinel = lc.Constant.struct([NULL, NULL, ZERO, NULL])
        method_defs.append(sentinel)
        method_array_init = lc.Constant.array(self.method_def_ty, method_defs)
        method_array = llvm_module.add_global_variable(method_array_init.type, '.module_methods')
        method_array.initializer = method_array_init
        method_array.linkage = lc.LINKAGE_INTERNAL
        method_array_ptr = lc.Constant.gep(method_array, [ZERO, ZERO])
        return method_array_ptr


class CompilerPy2(_Compiler):

    @property
    def module_create_definition(self):
        """Return the signature and name of the function to initialize the module.
        """
        signature = lc.Type.function(lt._pyobject_head_struct_p,
                                     (lt._int8_star,
                                      self.method_def_ptr,
                                      lt._int8_star,
                                      lt._pyobject_head_struct_p,
                                      lt._int32))

        name = "Py_InitModule4"

        if lt._trace_refs_:
            name += "TraceRefs"
        if lt._plat_bits == 64:
            name += "_64"

        return signature, name

    @property
    def module_init_definition(self):
        """Return the signature and name of the function to initialize the extension.
        """
        return lc.Type.function(lc.Type.void(), ()), "init" + self.module_name

    def _emit_python_wrapper(self, llvm_module):

        # Define the module initialization function.
        mod_init_fn = llvm_module.add_function(*self.module_init_definition)
        entry = mod_init_fn.append_basic_block('Entry')
        builder = lc.Builder.new(entry)

        # Python C API module creation function.
        create_module_fn = llvm_module.add_function(*self.module_create_definition)
        create_module_fn.linkage = lc.LINKAGE_EXTERNAL

        # Define a constant string for the module name.
        mod_name_init = lc.Constant.stringz(self.module_name)
        mod_name_const = llvm_module.add_global_variable(mod_name_init.type, '.module_name')
        mod_name_const.initializer = mod_name_init
        mod_name_const.linkage = lc.LINKAGE_INTERNAL

        mod = builder.call(create_module_fn,
                           (lc.Constant.gep(mod_name_const, [ZERO, ZERO]),
                            self._emit_method_array(llvm_module),
                            NULL,
                            lc.Constant.null(lt._pyobject_head_struct_p),
                            lc.Constant.int(lt._int32, sys.api_version)))

        builder.ret_void()


class CompilerPy3(_Compiler):

    _ptr_fun = lambda ret, *args: lc.Type.pointer(lc.Type.function(ret, args))

    #: typedef int (*visitproc)(PyObject *, void *);
    visitproc_ty = _ptr_fun(lt._int8,
                            lt._pyobject_head_struct_p)

    #: typedef int (*inquiry)(PyObject *);
    inquiry_ty = _ptr_fun(lt._int8,
                          lt._pyobject_head_struct_p)

    #: typedef int (*traverseproc)(PyObject *, visitproc, void *);
    traverseproc_ty = _ptr_fun(lt._int8,
                               lt._pyobject_head_struct_p,
                               visitproc_ty,
                               lt._void_star)

    #  typedef void (*freefunc)(void *)
    freefunc_ty = _ptr_fun(lt._int8,
                           lt._void_star)

    # PyObject* (*m_init)(void);
    m_init_ty = _ptr_fun(lt._int8)

    _char_star = lt._int8_star

    #: typedef struct PyModuleDef_Base {
    #:   PyObject_HEAD
    #:   PyObject* (*m_init)(void);
    #:   Py_ssize_t m_index;
    #:   PyObject* m_copy;
    #: } PyModuleDef_Base;
    module_def_base_ty = lc.Type.struct((lt._pyobject_head_struct_p,
                                         lt._void_star,
                                         m_init_ty,
                                         lt._llvm_py_ssize_t,
                                         lt._pyobject_head_struct_p))

    #: This struct holds all information that is needed to create a module object.
    #: typedef struct PyModuleDef{
    #:   PyModuleDef_Base m_base;
    #:   const char* m_name;
    #:   const char* m_doc;
    #:   Py_ssize_t m_size;
    #:   PyMethodDef *m_methods;
    #:   inquiry m_reload;
    #:   traverseproc m_traverse;
    #:   inquiry m_clear;
    #:   freefunc m_free;
    #: }PyModuleDef;
    module_def_ty = lc.Type.struct((module_def_base_ty,
                                    _char_star,
                                    _char_star,
                                    lt._llvm_py_ssize_t,
                                    _Compiler.method_def_ptr,
                                    inquiry_ty,
                                    traverseproc_ty,
                                    inquiry_ty,
                                    freefunc_ty))

    @property
    def module_create_definition(self):
        """Return the signature and name of the function to initialize the module
        """
        signature = lc.Type.function(lt._pyobject_head_struct_p,
                                     (lc.Type.pointer(self.module_def_ty),
                                      lt._int32))

        name = "PyModule_Create2"

        return signature, name

    @property
    def module_init_definition(self):
        """Return the name and signature of the module
        """
        signature = lc.Type.function(lt._pyobject_head_struct_p, ())

        return signature, "PyInit_" + self.module_name

    def _emit_python_wrapper(self, llvm_module):

        # Figure out the Python C API module creation function, and
        # get a LLVM function for it.
        create_module_fn = llvm_module.add_function(*self.module_create_definition)
        create_module_fn.linkage = lc.LINKAGE_EXTERNAL

        # Define a constant string for the module name.
        mod_name_init = lc.Constant.stringz(self.module_name)
        mod_name_const = llvm_module.add_global_variable(mod_name_init.type, '.module_name')
        mod_name_const.initializer = mod_name_init
        mod_name_const.linkage = lc.LINKAGE_INTERNAL

        mod_def_base_init = lc.Constant.struct(
            (lc.Constant.null(lt._pyobject_head_struct_p),  # PyObject_HEAD
             lc.Constant.null(lt._void_star),               # PyObject_HEAD
             lc.Constant.null(self.m_init_ty),              # m_init
             lc.Constant.null(lt._llvm_py_ssize_t),         # m_index
             lc.Constant.null(lt._pyobject_head_struct_p),  # m_copy
            )
        )
        mod_def_base = llvm_module.add_global_variable(mod_def_base_init.type, '.module_def_base')
        mod_def_base.initializer = mod_def_base_init
        mod_def_base.linkage = lc.LINKAGE_INTERNAL

        mod_def_init = lc.Constant.struct(
            (mod_def_base_init,                              # m_base
             lc.Constant.gep(mod_name_const, [ZERO, ZERO]),  # m_name
             lc.Constant.null(self._char_star),              # m_doc
             lc.Constant.int(lt._llvm_py_ssize_t, -1),       # m_size
             self._emit_method_array(llvm_module),           # m_methods
             lc.Constant.null(self.inquiry_ty),              # m_reload
             lc.Constant.null(self.traverseproc_ty),         # m_traverse
             lc.Constant.null(self.inquiry_ty),              # m_clear
             lc.Constant.null(self.freefunc_ty)              # m_free
            )
        )

        # Define a constant string for the module name.
        mod_def = llvm_module.add_global_variable(mod_def_init.type, '.module_def')
        mod_def.initializer = mod_def_init
        mod_def.linkage = lc.LINKAGE_INTERNAL

        # Define the module initialization function.
        mod_init_fn = llvm_module.add_function(*self.module_init_definition)
        entry = mod_init_fn.append_basic_block('Entry')
        builder = lc.Builder.new(entry)
        mod = builder.call(create_module_fn,
                           (mod_def,
                            lc.Constant.int(lt._int32, sys.api_version)))

        # Test if module has been created correctly
        cond_true = mod_init_fn.append_basic_block("cond_true")
        cond_false = mod_init_fn.append_basic_block("cond_false")
        builder.cbranch(builder.icmp(lc.IPRED_EQ, mod, NULL), cond_true, cond_false)
        builder.position_at_end(cond_true)
        builder.ret(NULL)

        builder.position_at_end(cond_false)
        builder.ret(mod)


Compiler = CompilerPy3 if IS_PY3 else CompilerPy2

