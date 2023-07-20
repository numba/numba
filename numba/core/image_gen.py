from abc import abstractmethod
from collections import namedtuple
from numba.core.codegen import Codegen, CodeLibrary, _check_llvm_bugs, dump,\
    get_host_cpu_features, initialize_llvm, _import_registered_symbols,\
    _parse_refprune_flags
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
import ctypes
import json
import llvmlite.binding as ll
import llvmlite.ir
import llvmlite.ir as llvmir
import numba.core.cgutils as cgutils
import numba.core.config as config
import os.path


def default_features():
    if config.CPU_FEATURES is not None:
        return config.CPU_FEATURES
    return get_host_cpu_features()


def _pass_manager_builder(**kwargs):
    """
    Create a PassManagerBuilder.

    Note: a PassManagerBuilder seems good only for one use, so you
    should call this method each time you want to populate a module
    or function pass manager.  Otherwise some optimizations will be
    missed...
    """
    opt_level = kwargs.pop('opt', config.OPT)
    loop_vectorize = kwargs.pop('loop_vectorize', config.LOOP_VECTORIZE)
    slp_vectorize = kwargs.pop('slp_vectorize', config.SLP_VECTORIZE)

    pmb = create_pass_manager_builder(opt=opt_level,
                                      loop_vectorize=loop_vectorize,
                                      slp_vectorize=slp_vectorize,
                                      **kwargs)

    return pmb


class BaseImageCodeLibrary(CodeLibrary):
    """
    A code library that compiles code to object code and allows a downstream
    process to process the object code
    """

    def __init__(self, codegen: "BaseImageCodegen", name, unique_name):
        super().__init__(codegen, name, unique_name)
        self._linking_libraries = []   # maintain insertion order
        self._llvm_ir = []
        self._globals = {}
        self._materialization_result = None
        self._native_asm = None

    def add_linking_library(self, library):
        self._raise_if_finalized()
        library._ensure_finalized()
        lib_name = library.unique_name
        if lib_name not in self._linking_libraries:
            self._linking_libraries.append(lib_name)

    def add_ir_module(self, ir_module):
        self._raise_if_finalized()
        assert isinstance(ir_module, llvmir.Module),\
            "Only llvmlite IR modules may be added"
        for var in ir_module.globals.values():
            if var.linkage in ('internal', 'linkonce', 'linkonce_odr'):
                continue
            if isinstance(var, llvmir.Function) and not var.basic_blocks:
                continue
            elif isinstance(var, llvmir.GlobalVariable) and not var.initializer:
                continue
            if var.name in self._globals:
                existing = self._globals[var.name]
                assert var.type == existing, \
                    ("Duplicate global " + var.name +
                     " has incompatible types " + var.type + " vs " +
                     existing + ".")
            else:
                self._globals[var.name] = var.type

        self._llvm_ir.append(cgutils.normalize_ir_text(str(ir_module)))

    def finalize(self):
        self._raise_if_finalized()
        self._finalized = True

        # Report any LLVM-related problems to the user
        _check_llvm_bugs()

        if config.DUMP_FUNC_OPT:
            dump("FUNCTION OPTIMIZED DUMP %s" % self.name,
                 self.get_llvm_str(), 'llvm')

        llcontext = ll.create_context()
        mod = None
        for ir in self._llvm_ir:
            accessory = ll.parse_assembly(ir, context=llcontext)
            # Enforce data layout to enable layout-specific optimizations
            accessory.data_layout = self._codegen._data_layout()

            with ll.create_function_pass_manager(accessory) as fpm:
                self._codegen._target_machine.add_analysis_passes(fpm)
                with _pass_manager_builder() as pmb:
                    pmb.populate(fpm)
                if config.LLVM_REFPRUNE_PASS:
                    fpm.add_refprune_pass(_parse_refprune_flags())
                # Run function-level optimizations to reduce memory usage and
                # improve module-level optimization.
                for func in accessory.functions:
                    k = f"Function passes on {func.name!r}"
                    with self._recorded_timings.record(k):
                        fpm.initialize()
                        fpm.run(func)
                        fpm.finalize()

            if mod is None:
                mod = accessory
            else:
                mod.link_in(accessory)

        assert mod

        cheap_name = "Module passes (cheap optimization for refprune)"
        with self._recorded_timings.record(cheap_name):
            # A cheaper optimisation pass is run first to try and get as many
            # refops into the same function as possible via inlining
            self._codegen._mpm_cheap.run(mod)
        # Refop pruning is then run on the heavily inlined function
        if not config.LLVM_REFPRUNE_PASS:
            mod = remove_redundant_nrt_refct(mod)
        full_name = "Module passes (full optimization)"
        with self._recorded_timings.record(full_name):
            # The full optimisation suite is then run on the refop pruned IR
            self._codegen._mpm_full.run(mod)

        mod.verify()

        # Scan for dynamic globals
        for gv in mod.global_variables:
            if gv.name.startswith('numba.dynamic.globals'):
                self._dynamic_globals.append(gv.name)

        self._native_asm = self._codegen._target_machine.emit_assembly(mod)
        object_code = self._codegen._target_machine.emit_object(mod)

        if config.DUMP_OPTIMIZED:
            dump("OPTIMIZED DUMP %s" % self.name,
                 '\n'.join(self.get_llvm_str()),
                 'llvm')

        if config.DUMP_ASSEMBLY:
            dump("ASSEMBLY %s" % self.name, self.get_asm_str(), 'asm')

        self._materialization_result = self._codegen._handle_image(
            self.name,
            self.unique_name,
            object_code,
            self._native_asm,
            self._linking_libraries,
            self._globals)

    def get_defined_functions(self):
        """
        Get all functions defined in the library.  The library must have
        been finalized.
        """
        self._ensure_finalized()
        for fn, fn_ty in self._globals:
            yield llvmlite.ir.Function(fn, None, fn_ty)

    def get_function(self, name):
        ty = self._globals[name]
        if isinstance(ty, llvmir.FunctionType):
            return llvmlite.ir.Function(name, None, ty)

    def get_llvm_str(self):
        return [*self._llvm_ir]

    def get_asm_str(self):
        return self._native_asm


ImageConfiguration = namedtuple('ImageConfiguration',
                                ('triple', 'cpu_name', 'features'))


class BaseImageCodegen(Codegen):

    def __init__(self, image_configuration):
        super().__init__()
        initialize_llvm()
        target = ll.Target.from_triple(image_configuration.triple)
        tm_options = dict(opt=config.OPT)
        if image_configuration.cpu_name == 'host':
            cpu_name = (ll.get_host_cpu_name()
                        if config.CPU_NAME is None
                        else config.CPU_NAME)
        else:
            cpu_name = image_configuration.cpu_name
        tm_options['cpu'] = cpu_name
        tm_options['reloc'] = 'pic'
        tm_options['codemodel'] = 'jitdefault'
        tm_options['features'] = image_configuration.features
        self._target_machine = target.create_target_machine(**tm_options)
        self._tm_features = image_configuration.features
        self._mpm_cheap = self._module_pass_manager(loop_vectorize=False,
                                                    slp_vectorize=False,
                                                    opt=0,
                                                    cost="cheap")
        self._mpm_full = self._module_pass_manager()

    @abstractmethod
    def _handle_image(self,
                      name,
                      unique_name,
                      object_code,
                      native_asm,
                      link_libraries,
                      global_values):
        """
        Loads or saves the generate assembly code
        """

    @abstractmethod
    def _data_layout(self):
        """
        Gets the data layout for modules associated with the code generator
        """

    def _create_empty_module(self, name):
        ir_module = llvmir.Module(cgutils.normalize_ir_text(name))
        ir_module.triple = ll.get_process_triple()
        ir_module.data_layout = self._data_layout()
        return ir_module

    def _module_pass_manager(self, **kwargs):
        pm = ll.create_module_pass_manager()
        self._target_machine.add_analysis_passes(pm)
        cost = kwargs.pop("cost", None)
        with _pass_manager_builder(**kwargs) as pmb:
            pmb.populate(pm)
        # If config.OPT==0 do not include these extra passes to help with
        # vectorization.
        if cost is not None and cost == "cheap" and config.OPT != 0:
            # This knocks loops into rotated form early to reduce the likelihood
            # of vectorization failing due to unknown PHI nodes.
            pm.add_loop_rotate_pass()
            if ll.llvm_version_info[0] < 12:
                # LLVM 11 added LFTR to the IV Simplification pass,
                # this interacted badly with the existing use of the
                # InstructionCombiner here and ended up with PHI nodes that
                # prevented vectorization from working. The desired
                # vectorization effects can be achieved with this in LLVM 11
                # (and also < 11) but at a potentially slightly higher cost:
                pm.add_licm_pass()
                pm.add_cfg_simplification_pass()
            else:
                # These passes are required to get SVML to vectorize tests
                # properly on LLVM 14
                pm.add_instruction_combining_pass()
                pm.add_jump_threading_pass()

        if config.LLVM_REFPRUNE_PASS:
            pm.add_refprune_pass(_parse_refprune_flags())
        return pm

    def magic_tuple(self):
        """
        Return a tuple unambiguously describing the codegen behaviour.
        """
        return self._target_machine.triple, self._tm_features

    def insert_unresolved_ref(self, builder, fnty, name):
        voidptr = llvmir.IntType(8).as_pointer()
        ptrname = self._rtlinker.PREFIX + name
        llvm_mod = builder.module
        try:
            fnptr = llvm_mod.get_function(ptrname)
        except KeyError:
            # Not defined?
            fnptr = llvmir.GlobalVariable(llvm_mod, voidptr, name=ptrname)
            fnptr.linkage = 'external'
        return builder.bitcast(builder.load(fnptr), fnty.as_pointer())


class OnDiskCodeLibrary(BaseImageCodeLibrary):
    pass


class OnDiskCodegen(BaseImageCodegen):
    """
    Create code on disk as object files, suitable for reloading later
    """

    _library_class = OnDiskCodeLibrary

    def __init__(self, image_configuration, storage_path, data_layout):
        super().__init__(image_configuration)
        self._storage_path = storage_path
        self._data_layout_value = str(data_layout)

    def _data_layout(self):
        return self._data_layout_value

    def _handle_image(self,
                      name,
                      unique_name,
                      object_code,
                      native_asm,
                      link_libraries,
                      global_values):
        def to_json(ty):
            if isinstance(ty, llvmir.FunctionType):
                return {"type": "function",
                        "return": to_json(ty.return_type),
                        "args": [to_json(a) for a in ty.args],
                        "variadic": ty.var_arg}
            elif isinstance(ty, llvmir.VoidType):
                return None
            elif isinstance(ty, llvmir.DoubleType):
                return "double"
            elif isinstance(ty, llvmir.FloatType):
                return "float"
            elif isinstance(ty, llvmir.HalfType):
                return "half"
            elif isinstance(ty, llvmir.IntType):
                return {"type": "int", "width": ty.width}
            elif isinstance(ty, llvmir.PointerType):
                return {"type": "ptr", "to": to_json(ty.pointee)}
            elif isinstance(ty, llvmir.ArrayType):
                return {"type": "array",
                        "element": to_json(ty.element),
                        "count": ty.count}
            elif isinstance(ty, llvmir.VectorType):
                return {"type": "vector",
                        "element": to_json(ty.element),
                        "count": ty.count}
            elif isinstance(ty, llvmir.LiteralStructType):
                return {"type": "literal_struct",
                        "elements": [to_json(e) for e in ty.elements],
                        "packed": ty.packed}
            elif isinstance(ty, llvmir.IdentifiedStructType):
                return {"type": "identified_struct",
                        "name": ty.name,
                        "elements": [to_json(e) for e in ty.elements],
                        "packed": ty.packed}
            else:
                raise ValueError("Cannot convert type to JSON")

        obj_path = os.path.join(self._storage_path, "%s.o" % unique_name)
        asm_path = os.path.join(self._storage_path, "%s.s" % unique_name)
        dep_path = os.path.join(self._storage_path, "%s.llinfo" % unique_name)
        with open(obj_path, "wb") as obj_file:
            obj_file.write(object_code)
        with open(asm_path, "wb") as asm_file:
            asm_file.write(native_asm)
        with open(dep_path, "w") as dep_file:
            json.dump({
                "name": name,
                "linking_libraries": link_libraries,
                "globals": {var_name: to_json(var_ty)
                            for var_name, var_ty in global_values}
            }, dep_file)
        return obj_path, dep_path

    def load_from_disk(self, unique_name):
        def from_json(ty):
            if ty is None:
                return llvmir.VoidType()
            elif ty == "double":
                return llvmir.DoubleType()
            elif ty == "float":
                return llvmir.FloatType()
            elif ty == "half":
                return llvmir.HalfType()
            if isinstance(ty, dict):
                if ty["type"] == "function":
                    return llvmir.FunctionType(from_json(ty["return"]),
                                               [from_json(a) for a in
                                                ty["args"]],
                                               var_arg=ty["variadic"])
                elif ty["type"] == "int":
                    return llvmir.IntType(ty["width"])
                elif ty["type"] == "ptr":
                    return from_json(ty["to"]).as_pointer()
                elif ty["type"] == "array":
                    return llvmir.ArrayType(from_json(ty["element"]),
                                            ty["count"])
                elif ty["type"] == "vector":
                    return llvmir.VectorType(from_json(ty["element"]),
                                             ty["count"])
                elif ty["type"] == "literal_struct":
                    return llvmir.LiteralStructType([from_json(e) for e
                                                     in ty["elements"]],
                                                    packed=ty["packed"])
                elif ty["type"] == "identified_struct":
                    return llvmir.IdentifiedStructType(ty["name"],
                                                       [from_json(e) for e
                                                        in ty["elements"]],
                                                       packed=ty["packed"])
            raise ValueError("Unknown JSON type descriptor")

        obj_path = os.path.join(self._storage_path, "%s.o" % unique_name)
        asm_path = os.path.join(self._storage_path, "%s.s" % unique_name)
        dep_path = os.path.join(self._storage_path, "%s.llinfo" % unique_name)
        assert os.path.exists(obj_path), f"Cannot find {obj_path}."
        with open(dep_path, "r") as dep_file:
            data = json.load(dep_file)
            lib = OnDiskCodeLibrary(self, data["name"], unique_name)
            lib._globals = {var_name: from_json(var_ty)
                            for var_name, var_ty in data["globals"]}
            lib._linking_libraries = data["linking_libraries"]
        lib._materialization_result = (obj_path, dep_path)
        with open(asm_path, "r") as asm_file:
            lib._native_asm = asm_file.read()
        return lib


class OrcJITCodeLibrary(BaseImageCodeLibrary):
    def get_pointer_to_function(self, name):
        self._ensure_finalized()
        return self._materialization_result[name]

    def set_env(self, env_name, env):
        """Set the environment address.

        Update the GlobalVariable named *env_name* to the address of *env*.
        """
        gvaddr = self._materialization_result[env_name]
        envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
        envptr[0] = ctypes.c_void_p(id(env))


class OrcJITCodegen(BaseImageCodegen):
    """
    Link code in memory using OrcJIT to allow functions to be callable
    """

    _library_class = OrcJITCodeLibrary
    _lljit = None

    def __init__(self):
        super().__init__(ImageConfiguration(triple=ll.get_process_triple(),
                                            cpu_name='host',
                                            features=default_features()))
        if not type(self)._lljit:
            type(self)._lljit = ll.create_lljit_compiler(self._target_machine)
        self._idgen = 0

    @property
    def _target_data(self):
        return type(self)._lljit.target_data

    def _data_layout(self):
        return str(type(self)._lljit.target_data)

    def _handle_image(self,
                      lib_name,
                      unique_name,
                      object_code,
                      native_asm,
                      link_libraries,
                      global_values):
        builder = ll.JITLibraryBuilder()
        builder.add_current_process()
        builder.add_object_img(object_code)
        _import_registered_symbols(builder)

        for dep in link_libraries:
            builder.add_jit_library(dep)

        for name in global_values.keys():
            builder.export_symbol(name)

        return builder.link(type(self)._lljit, unique_name)
