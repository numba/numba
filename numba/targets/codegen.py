from __future__ import print_function, division, absolute_import

import functools
import sys
import weakref

import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.passes as lp
import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba import config, utils
from numba.runtime.atomicops import remove_redundant_nrt_refct

_x86arch = frozenset(['x86', 'i386', 'i486', 'i586', 'i686', 'i786',
                      'i886', 'i986'])

def _is_x86(triple):
    arch = triple.split('-')[0]
    return arch in _x86arch


def dump(header, body):
    print(header.center(80, '-'))
    print(body)
    print('=' * 80)


class CodeLibrary(object):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """

    _finalized = False
    _object_caching_enabled = False

    def __init__(self, codegen, name):
        self._codegen = codegen
        self._name = name
        self._linking_libraries = set()
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self._name)))
        self._final_module.name = self._name
        # Remember this on the module, for the object cache hooks
        self._final_module.__library = weakref.proxy(self)
        self._shared_module = None

    @property
    def codegen(self):
        """
        The codegen object owning this library.
        """
        return self._codegen

    def __repr__(self):
        return "<Library %r at 0x%x>" % (self._name, id(self))

    def _raise_if_finalized(self):
        if self._finalized:
            raise RuntimeError("operation impossible on finalized object %r"
                               % (self,))

    def _ensure_finalized(self):
        if not self._finalized:
            self.finalize()

    def _optimize_functions(self, ll_module):
        """
        Internal: run function-level optimizations inside *ll_module*.
        """
        # Enforce data layout to enable layout-specific optimizations
        ll_module.data_layout = self._codegen._data_layout
        with self._codegen._function_pass_manager(ll_module) as fpm:
            # Run function-level optimizations to reduce memory usage and improve
            # module-level optimization.
            for func in ll_module.functions:
                fpm.initialize()
                fpm.run(func)
                fpm.finalize()

    def _optimize_final_module(self):
        """
        Internal: optimize this library's final module.
        """
        self._codegen._mpm.run(self._final_module)

    def _get_module_for_linking(self):
        """
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        """
        if self._shared_module is not None:
            return self._shared_module
        mod = self._final_module
        to_fix = []
        nfuncs = 0
        for fn in mod.functions:
            nfuncs += 1
            if not fn.is_declaration and fn.linkage == ll.Linkage.external:
                to_fix.append(fn.name)
        if nfuncs == 0:
            # This is an issue which can occur if loading a module
            # from an object file and trying to link with it, so detect it
            # here to make debugging easier.
            raise RuntimeError("library unfit for linking: "
                               "no available functions in %s"
                               % (self,))
        if to_fix:
            mod = mod.clone()
            for name in to_fix:
                # NOTE: this will mark the symbol WEAK if serialized
                # to an ELF file
                mod.get_function(name).linkage = 'linkonce_odr'
        self._shared_module = mod
        return mod

    def create_ir_module(self, name):
        """
        Create a LLVM IR module for use by this library.
        """
        self._raise_if_finalized()
        ir_module = self._codegen._create_empty_module(name)
        return ir_module

    def add_linking_library(self, library):
        """
        Add a library for linking into this library, without losing
        the original library.
        """
        library._ensure_finalized()
        self._linking_libraries.add(library)

    def add_ir_module(self, ir_module):
        """
        Add a LLVM IR module's contents to this library.
        """
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module)
        ll_module = ll.parse_assembly(str(ir_module))
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._optimize_functions(ll_module)
        # TODO: we shouldn't need to recreate the LLVM module object
        ll_module = remove_redundant_nrt_refct(ll_module)
        self._final_module.link_in(ll_module)

    def finalize(self):
        """
        Finalize the library.  After this call, nothing can be added anymore.
        Finalization involves various stages of code optimization and
        linking.
        """
        self._raise_if_finalized()

        if config.DUMP_FUNC_OPT:
            dump("FUNCTION OPTIMIZED DUMP %s" % self._name, self.get_llvm_str())

        # Link libraries for shared code
        for library in self._linking_libraries:
            self._final_module.link_in(
                library._get_module_for_linking(), preserve=True)
        for library in self._codegen._libraries:
            self._final_module.link_in(
                library._get_module_for_linking(), preserve=True)

        # Optimize the module after all dependences are linked in above,
        # to allow for inlining.
        self._optimize_final_module()

        self._final_module.verify()
        self._finalize_final_module()

    def _finalize_final_module(self):
        """
        Make the underlying LLVM module ready to use.
        """
        # It seems add_module() must be done only here and not before
        # linking in other modules, otherwise get_pointer_to_function()
        # could fail.
        cleanup = self._codegen._add_module(self._final_module)
        if cleanup:
            utils.finalize(self, cleanup)
        self._finalize_specific()

        self._finalized = True

        if config.DUMP_OPTIMIZED:
            dump("OPTIMIZED DUMP %s" % self._name, self.get_llvm_str())

        if config.DUMP_ASSEMBLY:
            # CUDA backend cannot return assembly this early, so don't
            # attempt to dump assembly if nothing is produced.
            asm = self.get_asm_str()
            if asm:
                dump("ASSEMBLY %s" % self._name, self.get_asm_str())

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        mod = self._final_module
        for fn in mod.functions:
            if not fn.is_declaration:
                yield fn

    def get_function(self, name):
        return self._final_module.get_function(name)

    def get_llvm_str(self):
        """
        Get the human-readable form of the LLVM module.
        """
        return str(self._final_module)

    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """
        return str(self._codegen._tm.emit_assembly(self._final_module))

    #
    # Object cache hooks and serialization
    #

    def enable_object_caching(self):
        self._object_caching_enabled = True
        self._compiled_object = None
        self._compiled = False

    def _get_compiled_object(self):
        if not self._object_caching_enabled:
            raise ValueError("object caching not enabled in %s" % (self,))
        if self._compiled_object is None:
            raise RuntimeError("no compiled object yet for %s" % (self,))
        return self._compiled_object

    def _set_compiled_object(self, value):
        if not self._object_caching_enabled:
            raise ValueError("object caching not enabled in %s" % (self,))
        if self._compiled:
            raise ValueError("library already compiled: %s" % (self,))
        self._compiled_object = value

    @classmethod
    def _dump_elf(cls, buf):
        """
        Dump the symbol table of an ELF file.
        Needs pyelftools (https://github.com/eliben/pyelftools)
        """
        from elftools.elf.elffile import ELFFile
        from elftools.elf import descriptions
        from io import BytesIO
        f = ELFFile(BytesIO(buf))
        print("ELF file:")
        for sec in f.iter_sections():
            if sec['sh_type'] == 'SHT_SYMTAB':
                symbols = sorted(sec.iter_symbols(), key=lambda sym: sym.name)
                print("    symbols:")
                for sym in symbols:
                    if not sym.name:
                        continue
                    print("    - %r: size=%d, value=0x%x, type=%s, bind=%s"
                          % (sym.name.decode(),
                             sym['st_size'],
                             sym['st_value'],
                             descriptions.describe_symbol_type(sym['st_info']['type']),
                             descriptions.describe_symbol_bind(sym['st_info']['bind']),
                             ))
        print()

    @classmethod
    def _object_compiled_hook(cls, ll_module, buf):
        """
        `ll_module` was compiled into object code `buf`.
        """
        try:
            self = ll_module.__library
        except AttributeError:
            return
        if self._object_caching_enabled:
            self._compiled = True
            self._compiled_object = buf

    @classmethod
    def _object_getbuffer_hook(cls, ll_module):
        """
        Return a cached object code for `ll_module`.
        """
        try:
            self = ll_module.__library
        except AttributeError:
            return
        if self._object_caching_enabled and self._compiled_object:
            buf = self._compiled_object
            self._compiled_object = None
            return buf

    def serialize_using_bitcode(self):
        """
        Serialize this library using its bitcode as the cached representation.
        """
        self._ensure_finalized()
        return (self._name, 'bitcode', self._final_module.as_bitcode())

    def serialize_using_object_code(self):
        """
        Serialize this library using its object code as the cached
        representation.
        """
        self._ensure_finalized()
        ll_module = self._final_module
        return (self._name, 'object', self._get_compiled_object())

    @classmethod
    def _unserialize(cls, codegen, state):
        name, kind, data = state
        self = codegen.create_library(name)
        assert isinstance(self, cls)
        if kind == 'bitcode':
            # No need to re-run optimizations, just make the module ready
            self._final_module = ll.parse_bitcode(data)
            self._finalize_final_module()
            return self
        elif kind == 'object':
            self.enable_object_caching()
            self._set_compiled_object(data)
            self._finalize_final_module()
            return self
        else:
            raise ValueError("unsupported serialization kind %r" % (kind,))


class AOTCodeLibrary(CodeLibrary):

    def emit_native_object(self):
        """
        Return this library as a native object (a bytestring) -- for example
        ELF under Linux.

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._codegen._tm.emit_object(self._final_module)

    def emit_bitcode(self):
        """
        Return this library as LLVM bitcode (a bytestring).

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._final_module.as_bitcode()

    def _finalize_specific(self):
        pass


class JITCodeLibrary(CodeLibrary):

    def get_pointer_to_function(self, name):
        """
        Generate native code for function named *name* and return a pointer
        to the start of the function (as an integer).

        This function implicitly calls .finalize().
        """
        self._ensure_finalized()
        return self._codegen._engine.get_function_address(name)

    def _finalize_specific(self):
        self._codegen._engine.finalize_object()


class BaseCPUCodegen(object):

    def __init__(self, module_name):
        self._libraries = set()
        self._data_layout = None
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._llvm_module.name = "global_codegen_module"
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"

        target = ll.Target.from_default_triple()
        tm_options = dict(cpu='', features='', opt=config.OPT)
        self._customize_tm_options(tm_options)
        tm = target.create_target_machine(**tm_options)
        engine = ll.create_mcjit_compiler(llvm_module, tm)
        tli = ll.create_target_library_info(llvm_module.triple)

        self._tli = tli
        self._tm = tm
        self._engine = engine
        self._target_data = engine.target_data
        self._data_layout = str(self._target_data)
        self._mpm = self._module_pass_manager()

        self._engine.set_object_cache(self._library_class._object_compiled_hook,
                                      self._library_class._object_getbuffer_hook)

    def _create_empty_module(self, name):
        ir_module = lc.Module.new(name)
        ir_module.triple = ll.get_default_triple()
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        return ir_module

    @property
    def target_data(self):
        """
        The LLVM "target data" object for this codegen instance.
        """
        return self._target_data

    def add_linking_library(self, library):
        """
        Add a library for linking into all libraries created by this
        codegen object, without losing the original library.
        """
        library._ensure_finalized()
        self._libraries.add(library)

    def create_library(self, name):
        """
        Create a :class:`CodeLibrary` object for use with this codegen
        instance.
        """
        return self._library_class(self, name)

    def unserialize_library(self, serialized):
        return self._library_class._unserialize(self, serialized)

    def _module_pass_manager(self):
        pm = ll.create_module_pass_manager()
        dl = ll.create_target_data(self._data_layout)
        dl.add_pass(pm)
        self._tli.add_pass(pm)
        self._tm.add_analysis_passes(pm)
        with self._pass_manager_builder() as pmb:
            pmb.populate(pm)
        return pm

    def _function_pass_manager(self, llvm_module):
        pm = ll.create_function_pass_manager(llvm_module)
        self._target_data.add_pass(pm)
        self._tli.add_pass(pm)
        self._tm.add_analysis_passes(pm)
        with self._pass_manager_builder() as pmb:
            pmb.populate(pm)
        return pm

    def _pass_manager_builder(self):
        """
        Create a PassManagerBuilder.

        Note: a PassManagerBuilder seems good only for one use, so you
        should call this method each time you want to populate a module
        or function pass manager.  Otherwise some optimizations will be
        missed...
        """
        pmb = lp.create_pass_manager_builder(
            opt=config.OPT, loop_vectorize=config.LOOP_VECTORIZE)
        return pmb

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        return (self._llvm_module.triple, ll.get_host_cpu_name(),
                config.ENABLE_AVX)


class AOTCPUCodegen(BaseCPUCodegen):
    """
    A codegen implementation suitable for Ahead-Of-Time compilation
    (e.g. generation of object files).
    """

    _library_class = AOTCodeLibrary

    def _customize_tm_options(self, options):
        options['reloc'] = 'pic'
        options['codemodel'] = 'default'

    def _add_module(self, module):
        pass


class JITCPUCodegen(BaseCPUCodegen):
    """
    A codegen implementation suitable for Just-In-Time compilation.
    """

    _library_class = JITCodeLibrary

    def _customize_tm_options(self, options):
        features = []

        # As long as we don't want to ship the code to another machine,
        # we can specialize for this CPU.
        options['cpu'] = ll.get_host_cpu_name()

        options['reloc'] = 'default'
        options['codemodel'] = 'jitdefault'

        # There are various performance issues with AVX and LLVM 3.5
        # (list at http://llvm.org/bugs/buglist.cgi?quicksearch=avx).
        # For now we'd rather disable it, since it can pessimize the code.
        if not config.ENABLE_AVX:
            features.append('-avx')

        # Set feature attributes
        options['features'] = ','.join(features)

        # Enable JIT debug
        options['jitdebug'] = True

    def _add_module(self, module):
        self._engine.add_module(module)
        # Early bind the engine method to avoid keeping a reference to self.
        return functools.partial(self._engine.remove_module, module)
