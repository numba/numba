from __future__ import print_function, division, absolute_import

import warnings
import functools
import locale
import weakref
import ctypes

import llvmlite.llvmpy.core as lc
import llvmlite.llvmpy.passes as lp
import llvmlite.binding as ll
import llvmlite.ir as llvmir

from numba import config, utils, cgutils
from numba.runtime.nrtopt import remove_redundant_nrt_refct
from numba.runtime import rtsys
from numba.compiler_lock import require_global_compiler_lock

_x86arch = frozenset(['x86', 'i386', 'i486', 'i586', 'i686', 'i786',
                      'i886', 'i986'])


def _is_x86(triple):
    arch = triple.split('-')[0]
    return arch in _x86arch


def dump(header, body):
    print(header.center(80, '-'))
    print(body)
    print('=' * 80)


class _CFG(object):
    """
    Wraps the CFG graph for different display method.

    Instance of the class can be stringified (``__repr__`` is defined) to get
    the graph in DOT format.  The ``.display()`` method plots the graph in
    PDF.  If in IPython notebook, the returned image can be inlined.
    """
    def __init__(self, dot):
        self.dot = dot

    def display(self, filename=None, view=False):
        """
        Plot the CFG.  In IPython notebook, the return image object can be
        inlined.

        The *filename* option can be set to a specific path for the rendered
        output to write to.  If *view* option is True, the plot is opened by
        the system default application for the image format (PDF).
        """
        return ll.view_dot_graph(self.dot, filename=filename, view=view)

    def __repr__(self):
        return self.dot


class CodeLibrary(object):
    """
    An interface for bundling LLVM code together and compiling it.
    It is tied to a *codegen* instance (e.g. JITCPUCodegen) that will
    determine how the LLVM code is transformed and linked together.
    """

    _finalized = False
    _object_caching_enabled = False
    _disable_inspection = False

    def __init__(self, codegen, name):
        self._codegen = codegen
        self._name = name
        self._linking_libraries = set()
        self._final_module = ll.parse_assembly(
            str(self._codegen._create_empty_module(self._name)))
        self._final_module.name = cgutils.normalize_ir_text(self._name)
        self._shared_module = None
        # Track names of the dynamic globals
        self._dynamic_globals = []

    @property
    def has_dynamic_globals(self):
        return len(self._dynamic_globals) > 0

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
        self._final_module = remove_redundant_nrt_refct(self._final_module)

    def _get_module_for_linking(self):
        """
        Internal: get a LLVM module suitable for linking multiple times
        into another library.  Exported functions are made "linkonce_odr"
        to allow for multiple definitions, inlining, and removal of
        unused exports.

        See discussion in https://github.com/numba/numba/pull/890
        """
        self._ensure_finalized()
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
        ir = cgutils.normalize_ir_text(str(ir_module))
        ll_module = ll.parse_assembly(ir)
        ll_module.name = ir_module.name
        ll_module.verify()
        self.add_llvm_module(ll_module)

    def _scan_dynamic_globals(self, ll_module):
        """
        Scan for dynanmic globals and track their names
        """
        for gv in ll_module.global_variables:
            if gv.name.startswith("numba.dynamic.globals"):
                self._dynamic_globals.append(gv.name)

    def add_llvm_module(self, ll_module):
        self._scan_dynamic_globals(ll_module)
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
        require_global_compiler_lock()

        # Report any LLVM-related problems to the user
        self._codegen._check_llvm_bugs()

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
        # Remember this on the module, for the object cache hooks
        self._final_module.__library = weakref.proxy(self)

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

    def _sentry_cache_disable_inspection(self):
        if self._disable_inspection:
            warnings.warn('Inspection disabled for cached code. '
                          'Invalid result is returned.')

    def get_llvm_str(self):
        """
        Get the human-readable form of the LLVM module.
        """
        self._sentry_cache_disable_inspection()
        return str(self._final_module)

    def get_asm_str(self):
        """
        Get the human-readable assembly.
        """
        self._sentry_cache_disable_inspection()
        return str(self._codegen._tm.emit_assembly(self._final_module))

    def get_function_cfg(self, name):
        """
        Get control-flow graph of the LLVM function
        """
        self._sentry_cache_disable_inspection()
        fn = self.get_function(name)
        dot = ll.get_function_cfg(fn)
        return _CFG(dot)

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
        self._disable_inspection = True

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
        representation.  We also include its bitcode for further inlining
        with other libraries.
        """
        self._ensure_finalized()
        data = (self._get_compiled_object(),
                self._get_module_for_linking().as_bitcode())
        return (self._name, 'object', data)

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
            object_code, shared_bitcode = data
            self.enable_object_caching()
            self._set_compiled_object(object_code)
            self._shared_module = ll.parse_bitcode(shared_bitcode)
            self._finalize_final_module()
            # Load symbols from cache
            self._codegen._engine._load_defined_symbols(self._shared_module)
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

        Returns
        -------
        pointer : int
            - zero (null) if no symbol of *name* is defined by this code
              library.
            - non-zero if the symbol is defined.
        """
        self._ensure_finalized()
        ee = self._codegen._engine
        if not ee.is_symbol_defined(name):
            return 0
        else:
            return self._codegen._engine.get_function_address(name)

    def _finalize_specific(self):
        self._codegen._scan_and_fix_unresolved_refs(self._final_module)
        self._codegen._engine.finalize_object()


class RuntimeLinker(object):
    """
    For tracking unresolved symbols generated at runtime due to recursion.
    """
    PREFIX = '.numba.unresolved$'

    def __init__(self):
        self._unresolved = utils.UniqueDict()
        self._defined = set()
        self._resolved = []

    def scan_unresolved_symbols(self, module, engine):
        """
        Scan and track all unresolved external symbols in the module and
        allocate memory for it.
        """
        prefix = self.PREFIX

        for gv in module.global_variables:
            if gv.name.startswith(prefix):
                sym = gv.name[len(prefix):]
                # Avoid remapping to existing GV
                if engine.is_symbol_defined(gv.name):
                    continue
                # Allocate a memory space for the pointer
                abortfn = rtsys.library.get_pointer_to_function("nrt_unresolved_abort")
                ptr = ctypes.c_void_p(abortfn)
                engine.add_global_mapping(gv, ctypes.addressof(ptr))
                self._unresolved[sym] = ptr

    def scan_defined_symbols(self, module):
        """
        Scan and track all defined symbols.
        """
        for fn in module.functions:
            if not fn.is_declaration:
                self._defined.add(fn.name)

    def resolve(self, engine):
        """
        Fix unresolved symbols if they are defined.
        """
        # An iterator to get all unresolved but available symbols
        pending = [name for name in self._unresolved if name in self._defined]
        # Resolve pending symbols
        for name in pending:
            # Get runtime address
            fnptr = engine.get_function_address(name)
            # Fix all usage
            ptr = self._unresolved[name]
            ptr.value = fnptr
            self._resolved.append((name, ptr))   # keep ptr alive
            # Delete resolved
            del self._unresolved[name]


def _proxy(old):
    @functools.wraps(old)
    def wrapper(self, *args, **kwargs):
        return old(self._ee, *args, **kwargs)
    return wrapper


class JitEngine(object):
    """Wraps an ExecutionEngine to provide custom symbol tracking.
    Since the symbol tracking is incomplete  (doesn't consider
    loaded code object), we are not putting it in llvmlite.
    """
    def __init__(self, ee):
        self._ee = ee
        # Track symbol defined via codegen'd Module
        # but not any cached object.
        # NOTE: `llvm::ExecutionEngine` will catch duplicated symbols and
        # we are not going to protect against that.  A proper duplicated
        # symbol detection will need a more logic to check for the linkage
        # (e.g. like `weak` linkage symbol can override).   This
        # `_defined_symbols` set will be just enough to tell if a symbol
        # exists and will not cause the `EE` symbol lookup to `exit(1)`
        # when symbol-not-found.
        self._defined_symbols = set()

    def is_symbol_defined(self, name):
        """Is the symbol defined in this session?
        """
        return name in self._defined_symbols

    def _load_defined_symbols(self, mod):
        """Extract symbols from the module
        """
        for gsets in (mod.functions, mod.global_variables):
            self._defined_symbols |= {gv.name for gv in gsets
                                      if not gv.is_declaration}

    def add_module(self, module):
        """Override ExecutionEngine.add_module
        to keep info about defined symbols.
        """
        self._load_defined_symbols(module)
        return self._ee.add_module(module)

    def add_global_mapping(self, gv, addr):
        """Override ExecutionEngine.add_global_mapping
        to keep info about defined symbols.
        """
        self._defined_symbols.add(gv.name)
        return self._ee.add_global_mapping(gv, addr)

    #
    # The remaining methods are re-export of the ExecutionEngine APIs
    #
    set_object_cache = _proxy(ll.ExecutionEngine.set_object_cache)
    finalize_object = _proxy(ll.ExecutionEngine.finalize_object)
    get_function_address = _proxy(ll.ExecutionEngine.get_function_address)
    get_global_value_address = _proxy(
        ll.ExecutionEngine.get_global_value_address
        )

class BaseCPUCodegen(object):

    def __init__(self, module_name):
        initialize_llvm()

        self._libraries = set()
        self._data_layout = None
        self._llvm_module = ll.parse_assembly(
            str(self._create_empty_module(module_name)))
        self._llvm_module.name = "global_codegen_module"
        self._rtlinker = RuntimeLinker()
        self._init(self._llvm_module)

    def _init(self, llvm_module):
        assert list(llvm_module.global_variables) == [], "Module isn't empty"

        target = ll.Target.from_triple(ll.get_process_triple())
        tm_options = dict(opt=config.OPT)
        self._tm_features = self._customize_tm_features()
        self._customize_tm_options(tm_options)
        tm = target.create_target_machine(**tm_options)
        engine = ll.create_mcjit_compiler(llvm_module, tm)

        if config.ENABLE_PROFILING:
            engine.enable_jit_events()

        self._tm = tm
        self._engine = JitEngine(engine)
        self._target_data = engine.target_data
        self._data_layout = str(self._target_data)
        self._mpm = self._module_pass_manager()

        self._engine.set_object_cache(self._library_class._object_compiled_hook,
                                      self._library_class._object_getbuffer_hook)

    def _create_empty_module(self, name):
        ir_module = lc.Module(cgutils.normalize_ir_text(name))
        ir_module.triple = ll.get_process_triple()
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
        self._tm.add_analysis_passes(pm)
        with self._pass_manager_builder() as pmb:
            pmb.populate(pm)
        return pm

    def _function_pass_manager(self, llvm_module):
        pm = ll.create_function_pass_manager(llvm_module)
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

    def _check_llvm_bugs(self):
        """
        Guard against some well-known LLVM bug(s).
        """
        # Check the locale bug at https://github.com/numba/numba/issues/1569
        # Note we can't cache the result as locale settings can change
        # accross a process's lifetime.  Also, for this same reason,
        # the check here is a mere heuristic (there may be a race condition
        # between now and actually compiling IR).
        ir = """
            define double @func()
            {
                ret double 1.23e+01
            }
            """
        mod = ll.parse_assembly(ir)
        ir_out = str(mod)
        if "12.3" in ir_out or "1.23" in ir_out:
            # Everything ok
            return
        if "1.0" in ir_out:
            loc = locale.getlocale()
            raise RuntimeError(
                "LLVM will produce incorrect floating-point code "
                "in the current locale %s.\nPlease read "
                "http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug "
                "for more information."
                % (loc,))
        raise AssertionError("Unexpected IR:\n%s\n" % (ir_out,))

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        return (self._llvm_module.triple, self._get_host_cpu_name(),
                self._tm_features)

    def _scan_and_fix_unresolved_refs(self, module):
        self._rtlinker.scan_unresolved_symbols(module, self._engine)
        self._rtlinker.scan_defined_symbols(module)
        self._rtlinker.resolve(self._engine)

    def insert_unresolved_ref(self, builder, fnty, name):
        voidptr = llvmir.IntType(8).as_pointer()
        ptrname = self._rtlinker.PREFIX + name
        llvm_mod = builder.module
        try:
            fnptr = llvm_mod.get_global(ptrname)
        except KeyError:
            # Not defined?
            fnptr = llvmir.GlobalVariable(llvm_mod, voidptr, name=ptrname)
            fnptr.linkage = 'external'
        return builder.bitcast(builder.load(fnptr), fnty.as_pointer())

    def _get_host_cpu_name(self):
        return (ll.get_host_cpu_name()
                if config.CPU_NAME is None
                else config.CPU_NAME)

    def _get_host_cpu_features(self):
        if config.CPU_FEATURES is not None:
            return config.CPU_FEATURES
        return get_host_cpu_features()

class AOTCPUCodegen(BaseCPUCodegen):
    """
    A codegen implementation suitable for Ahead-Of-Time compilation
    (e.g. generation of object files).
    """

    _library_class = AOTCodeLibrary

    def __init__(self, module_name, cpu_name=None):
        # By default, use generic cpu model for the arch
        self._cpu_name = cpu_name or ''
        BaseCPUCodegen.__init__(self, module_name)

    def _customize_tm_options(self, options):
        cpu_name = self._cpu_name
        if cpu_name == 'host':
            cpu_name = self._get_host_cpu_name()
        options['cpu'] = cpu_name
        options['reloc'] = 'pic'
        options['codemodel'] = 'default'
        options['features'] = self._tm_features

    def _customize_tm_features(self):
        # ISA features are selected according to the requested CPU model
        # in _customize_tm_options()
        return ''

    def _add_module(self, module):
        pass


class JITCPUCodegen(BaseCPUCodegen):
    """
    A codegen implementation suitable for Just-In-Time compilation.
    """

    _library_class = JITCodeLibrary

    def _customize_tm_options(self, options):
        # As long as we don't want to ship the code to another machine,
        # we can specialize for this CPU.
        options['cpu'] = self._get_host_cpu_name()
        options['reloc'] = 'default'
        options['codemodel'] = 'jitdefault'

        # Set feature attributes (such as ISA extensions)
        # This overrides default feature selection by CPU model above
        options['features'] = self._tm_features

        # Enable JIT debug
        options['jitdebug'] = True

    def _customize_tm_features(self):
        # For JIT target, we will use LLVM to get the feature map
        return self._get_host_cpu_features()

    def _add_module(self, module):
        self._engine.add_module(module)
        # XXX: disabling remove module due to MCJIT engine leakage in
        #      removeModule.  The removeModule causes consistent access
        #      violation with certain test combinations.
        # # Early bind the engine method to avoid keeping a reference to self.
        # return functools.partial(self._engine.remove_module, module)

    def set_env(self, env_name, env):
        """Set the environment address.

        Update the GlobalVariable named *env_name* to the address of *env*.
        """
        gvaddr = self._engine.get_global_value_address(env_name)
        envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
        envptr[0] = ctypes.c_void_p(id(env))


def initialize_llvm():
    """Safe to use multiple times.
    """
    ll.initialize()
    ll.initialize_native_target()
    ll.initialize_native_asmprinter()


def get_host_cpu_features():
    """Get host CPU features using LLVM.

    The features may be modified due to user setting.
    See numba.config.ENABLE_AVX.
    """
    try:
        features = ll.get_host_cpu_features()
    except RuntimeError:
        return ''
    else:
        if not config.ENABLE_AVX:
            # Disable all features with name starting with 'avx'
            for k in features:
                if k.startswith('avx'):
                    features[k] = False

        # Set feature attributes
        return features.flatten()
