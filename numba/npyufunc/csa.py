from numba import compiler, sigutils, types, dispatcher
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from numba import typing, config
from numba.targets.base import BaseContext
from numba.targets import csa as tcsa
from numba.targets import registry
from numba.ir_utils import get_call_table
from numba.npyufunc import dufunc

import pdb
import copy

class CSATypingContext(typing.BaseContext):
    def load_additional_registries(self):
        pass

    def resolve_value_type(self, val):
        if isinstance(val, dispatcher.Dispatcher):
            try:
                val = val.__csajitdevice
            except AttributeError:
                if not val._can_compile:
                    raise ValueError("using cpu function on device but its compilation is disabled")
                jd = jitdevice(val, debug=val.targetoptions.get('debug'))
                val.__csajitdevice = jd
                val = jd
        return super(CSATypingContext, self).resolve_value_type(val)

    def load_additional_registries(self):
        from numba.typing import (cffi_utils, cmathdecl, enumdecl, listdecl, mathdecl,
                       npydecl, operatordecl, randomdecl, setdecl)
        self.install_registry(cffi_utils.registry)
        self.install_registry(cmathdecl.registry)
        self.install_registry(enumdecl.registry)
        self.install_registry(listdecl.registry)
        self.install_registry(mathdecl.registry)
        self.install_registry(npydecl.registry)
        self.install_registry(operatordecl.registry)
        self.install_registry(randomdecl.registry)
        self.install_registry(setdecl.registry)

class CSATargetOptions(TargetOptions):
    OPTIONS = {}

class CSATargetDesc(TargetDescriptor):
    options = CSATargetOptions
    typingctx = CSATypingContext()
    targetctx = tcsa.CSAContext(typingctx)

def compile_csa(func_ir, return_type, args, inflags):
    if config.DEBUG_CSA:
        print("compile_csa", func_ir, return_type, args)

    cput = registry.dispatcher_registry['cpu'].targetdescr 
    typingctx = cput.typing_context
#    typingctx = CSATargetDesc.typingctx
    targetctx = CSATargetDesc.targetctx
    flags = compiler.Flags()
    #flags = copy.copy(inflags)
    if inflags.noalias:
        if config.DEBUG_CSA:
            print("Propagating noalias")
        flags.set('noalias')
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')

    (call_table, _) = get_call_table(func_ir.blocks) 
    if config.DEBUG_CSA:
        print("call_table", call_table)
    for key, value in call_table.items():
        if len(value) == 1:
            value = value[0]
            if isinstance(value, dufunc.DUFunc):
                if config.DEBUG_CSA:
                    print("Calling install_cg on", value)
                value._install_cg(targetctx)

    #pdb.set_trace()
    cres = compiler.compile_ir(typingctx,
                               targetctx,
                               func_ir,
                               args,
                               return_type,
                               flags,
                               locals={})
    library = cres.library
    library.finalize()

    if config.DEBUG_CSA:
        print("compile_csa cres", cres, type(cres))
        print("LLVM")

        llvm_str = cres.library.get_llvm_str()
        llvm_out = "compile_csa" + ".ll"
        print(llvm_out, llvm_str)
        with open(llvm_out, "w") as llvm_file:
            llvm_file.write(llvm_str)

    return cres

#class CachedCSAAsm(object):
#    def __init__(self, name, llvmir, options):
#        print("CachedCSAAsm::__init__", name, llvmir, options)
#        self.name = name
#        self.llvmir = llvmir
#        self._extra_options = options.copy()
#
#    def get(self):
#        print("CachedCSAAsm", self.name)
#        print(self.llvmir)
#        targetctx = CSATargetDesc.targetctx
#        print("targetctx", targetctx, type(targetctx))
#        cg = targetctx.codegen()
#        print("cg", cg, type(cg))
##        lib = cg.create_library(self.name)
##        csa_asm_name = self.name + '.csa_asm.s'
##        asm_str = lib.get_asm_str(csa_asm_name)
#        return csa_asm_name
#
#class CachedCSACUFunction(object):
#    def __init__(self, entry_name, csa_asm, linking):
#        self.entry_name = entry_name
#        self.csa_asm = csa_asm
#        self.linking = linking
# 
#    def get(self):
#        csa_asm = self.csa_asm.get()
#        return csa_asm

class CSAKernel(object):
    def __init__(self, llvm_module, library, wrapper_module, kernel, wrapfnty,
                 name, pretty_name, argtypes, call_helper,
                 link=(), debug=False, fastmath=False, type_annotation=None):
        options = {'debug': debug}
        if fastmath:
            options.update(dict(ftz=True,
                                prec_sqrt=False,
                                prec_div=False,
                                fma=True))

#        csa_asm = CachedCSAAsm(pretty_name, str(llvm_module), options=options)
#        cufunc = CachedCSACUFunction(name, csa_asm, link)
        self.kernel = kernel
        self.wrapfnty = wrapfnty
        # populate members
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.linking = tuple(link)
        self._type_annotation = type_annotation
#        self._func = cufunc
        self.debug = debug
        self.call_helper = call_helper
        self.llvm_module = llvm_module
        self.wrapper_module = wrapper_module
        self.library = library

    def __repr__(self):
        ret = "CSAKernel object\nself.kernel\n" + str(self.kernel) + str(type(self.kernel))
        ret += "\nself.library " + str(self.library) + " type=" + str(type(self.library))
        ret += "\nself.wrapfnty " + str(self.wrapfnty) + " type=" + str(type(self.wrapfnty))
        ret += "\nself.entry_name " + str(self.entry_name) + " type=" + str(type(self.entry_name))
        ret += "\nself.argument_types " + str(self.argument_types) + " type=" + str(type(self.argument_types))
        return ret

#    def __call__(self, *args, **kwargs):
#        assert not kwargs
#        self._kernel_call(args=args,
#                          griddim=self.griddim,
#                          blockdim=self.blockdim,
#                          stream=self.stream,
#                          sharedmem=self.sharedmem)

#    def bind(self):
#        print("self._func", type(self._func))
#        self._func.get()

#    @property
#    def csa_asm(self):
#        return self._func.csa_asm.get().decode('utf8')

def compile_csa_kernel(func_ir, args, flags, link, fastmath=False):
    cres = compile_csa(func_ir, types.void, args, flags)
    fname = cres.fndesc.llvm_func_name
    lib, kernel, wrapfnty, wrapper_library = cres.target_context.prepare_csa_kernel(cres.library, fname, cres.signature.args)
    if config.DEBUG_CSA:
        print("compile_csa_kernel", func_ir, args, link, fastmath)
        print("fname", fname, type(fname))
        print("cres.library", cres.library, type(cres.library))
        print("cres.signature", cres.signature, type(cres.signature))
        print("lib", lib, type(lib))
        print("kernel", kernel, type(kernel))
        print("wrapfnty", wrapfnty, type(wrapfnty))
    csakern = CSAKernel(llvm_module=lib._final_module,
                        library=wrapper_library,
                        wrapper_module=wrapper_library._final_module,
                        kernel=kernel, wrapfnty=wrapfnty, name=kernel.name,
                        pretty_name=cres.fndesc.qualname,
                        argtypes=args,
                        type_annotation=cres.type_annotation,
                        link=link,
                        debug=False,
                        call_helper=cres.call_helper,
                        fastmath=fastmath)
    return csakern

class AutoJitCSAKernel(object):
    def __init__(self, func_ir, bind, flags, targetoptions):
        self.func_ir = func_ir
        self.bind = bind
        self.flags = flags
        self.targetoptions = targetoptions
        self.typingctx = CSATargetDesc.typingctx
        self.definitions = {}

    def __call__(self, *args):
        kernel = self.specialize(*args)
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([self.typingctx.resolve_argument_type(a) for a in args])
        return self.compile(argtypes)

    def compile(self, sig):
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None
        kernel = self.definitions.get(argtypes)
        if kernel is None:
            if 'link' not in self.targetoptions:
                self.targetoptions['link'] = ()
            kernel = compile_csa_kernel(self.func_ir, argtypes, self.flags,
                                    **self.targetoptions)
            self.definitions[argtypes] = kernel
#            if self.bind:
#                kernel.bind()
        return kernel

    def inspect_llvm(self, signature=None):
        if signature is not None:
            return self.definitions[signature].inspect_llvm()
        else:
            return dict((sig, defn.inspect_llvm())
                        for sig, defn in self.definitions.items())

    def inspect_asm(self, signature=None):
        '''
        Return the generated assembly code for all signatures encountered thus
        far, or the LLVM IR for a specific signature if given.
        '''
        if signature is not None:
            return self.definitions[signature].inspect_asm()
        else:
            return dict((sig, defn.inspect_asm())
                        for sig, defn in self.definitions.items())

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if file is None:
            file = sys.stdout

        for ver, defn in utils.iteritems(self.definitions):
            defn.inspect_types(file=file)
