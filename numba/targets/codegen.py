from __future__ import print_function, division, absolute_import

import sys

import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.passes as lp
import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba import config


_x86arch = frozenset(['x86', 'i386', 'i486', 'i586', 'i686', 'i786',
                      'i886', 'i986'])

def _is_x86(triple):
    arch = triple.split('-')[0]
    return arch in _x86arch


class CodeLibrary(object):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """

    _finalized = False

    def __init__(self, codegen, name):
        self._codegen = codegen
        self._name = name
        self._linking_libraries = set()
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self._name)))
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
        for fn in mod.functions:
            if not fn.is_declaration and fn.linkage == ll.Linkage.external:
                to_fix.append(fn.name)
        if to_fix:
            mod = mod.clone()
            for name in to_fix:
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
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def add_llvm_module(self, ll_module):
        self._optimize_functions(ll_module)
        self._final_module.link_in(ll_module)

    def finalize(self):
        """
        Finalize the library.  After this call, nothing can be added anymore.
        Finalization involves various stages of code optimization and
        linking.
        """
        self._raise_if_finalized()

        if config.DUMP_FUNC_OPT:
            print(("FUNCTION OPTIMIZED DUMP %s" % self._name).center(80, '-'))
            print(self._final_module)
            print('=' * 80)

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
        # It seems add_module() must be done only here and not before
        # linking in other modules, otherwise get_pointer_to_function()
        # could fail.
        self._codegen._add_module(self._final_module)
        self._finalize_specific()

        self._finalized = True

        if config.DUMP_OPTIMIZED:
            print(("OPTIMIZED DUMP %s" % self._name).center(80, '-'))
            print(self._final_module)
            print('=' * 80)

        if config.DUMP_ASSEMBLY:
            self._dump_assembly()

    def get_function(self, name):
        return self._final_module.get_function(name)

    def _dump_assembly(self):
        """
        Internal: dump native assembler code for this library.
        """
        print(("ASSEMBLY %s" % self._name).center(80, '-'))
        print(self._codegen._tm.emit_assembly(self._final_module))
        print('=' * 80)


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
        func = self.get_function(name)
        return self._codegen._engine.get_pointer_to_function(func)

    def _finalize_specific(self):
        self._codegen._engine.finalize_object()


class BaseCPUCodegen(object):

    def __init__(self, module_name):
        self._libraries = set()
        self._data_layout = None
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"

        target = ll.Target.from_default_triple()
        tm_options = dict(cpu='', features='', opt=config.OPT)
        self._customize_tm_options(tm_options)
        tm = target.create_target_machine(**tm_options)

        # MCJIT is still defective under Windows
        if sys.platform.startswith('win32'):
            engine = ll.create_jit_compiler_with_tm(llvm_module, tm)
        else:
            engine = ll.create_mcjit_compiler(llvm_module, tm)

        tli = ll.create_target_library_info(llvm_module.triple)

        self._tli = tli
        self._tm = tm
        self._engine = engine
        self._target_data = engine.target_data
        self._data_layout = str(self._target_data)
        self._mpm = self._module_pass_manager()

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
