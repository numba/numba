from llvmlite import binding as ll
from llvmlite import ir

from numba.core import config, serialize
from numba.core.codegen import Codegen, CodeLibrary
from .cudadrv import devices, driver, nvvm

import ctypes
import numpy as np
import os
import subprocess
import tempfile


CUDA_TRIPLE = 'nvptx64-nvidia-cuda'


def disassemble_cubin(cubin):
    # nvdisasm only accepts input from a file, so we need to write out to a
    # temp file and clean up afterwards.
    fd = None
    fname = None
    try:
        fd, fname = tempfile.mkstemp()
        with open(fname, 'wb') as f:
            f.write(cubin)

        try:
            cp = subprocess.run(['nvdisasm', fname], check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        except FileNotFoundError as e:
            if e.filename == 'nvdisasm':
                msg = ("nvdisasm is required for SASS inspection, and has not "
                       "been found.\n\nYou may need to install the CUDA "
                       "toolkit and ensure that it is available on your "
                       "PATH.\n")
                raise RuntimeError(msg)
        return cp.stdout.decode('utf-8')
    finally:
        if fd is not None:
            os.close(fd)
        if fname is not None:
            os.unlink(fname)


class CUDACodeLibrary(serialize.ReduceMixin, CodeLibrary):
    """
    The CUDACodeLibrary generates PTX, SASS, cubins for multiple different
    compute capabilities. It also loads cubins to multiple devices (via
    get_cufunc), which may be of different compute capabilities.
    """

    def __init__(self, codegen, name, entry_name=None, max_registers=None,
                 nvvm_options=None):
        """
        codegen:
            Codegen object.
        name:
            Name of the function in the source.
        entry_name:
            Name of the kernel function in the binary, if this is a global
            kernel and not a device function.
        max_registers:
            The maximum register usage to aim for when linking.
        nvvm_options:
                Dict of options to pass to NVVM.
        """
        super().__init__(codegen, name)

        # The llvmlite module for this library.
        self._module = None
        # CodeLibrary objects that will be "linked" into this library. The
        # modules within them are compiled from NVVM IR to PTX along with the
        # IR from this module - in that sense they are "linked" by NVVM at PTX
        # generation time, rather than at link time.
        self._linking_libraries = set()
        # Files to link with the generated PTX. These are linked using the
        # Driver API at link time.
        self._linking_files = set()

        # Maps CC -> PTX string
        self._ptx_cache = {}
        # Maps CC -> cubin
        self._cubin_cache = {}
        # Maps CC -> linker info output for cubin
        self._linkerinfo_cache = {}
        # Maps Device numeric ID -> cufunc
        self._cufunc_cache = {}

        self._max_registers = max_registers
        if nvvm_options is None:
            nvvm_options = {}
        self._nvvm_options = nvvm_options
        self._entry_name = entry_name

    def get_llvm_str(self):
        return str(self._module)

    def get_asm_str(self, cc=None):
        if not cc:
            ctx = devices.get_context()
            device = ctx.device
            cc = device.compute_capability

        ptx = self._ptx_cache.get(cc, None)
        if ptx:
            return ptx

        arch = nvvm.get_arch_option(*cc)
        options = self._nvvm_options.copy()
        options['arch'] = arch

        irs = [str(mod) for mod in self.modules]
        ptx = nvvm.llvm_to_ptx(irs, **options)
        ptx = ptx.decode().strip('\x00').strip()

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % self._name).center(80, '-'))
            print(ptx)
            print('=' * 80)

        self._ptx_cache[cc] = ptx

        return ptx

    def get_cubin(self, cc=None):
        if cc is None:
            ctx = devices.get_context()
            device = ctx.device
            cc = device.compute_capability

        cubin = self._cubin_cache.get(cc, None)
        if cubin:
            return cubin

        ptx = self.get_asm_str(cc=cc)
        linker = driver.Linker(max_registers=self._max_registers, cc=cc)
        linker.add_ptx(ptx.encode())
        for path in self._linking_files:
            linker.add_file_guess_ext(path)
        cubin_buf, size = linker.complete()

        # We take a copy of the cubin because it's owned by the linker
        cubin_ptr = ctypes.cast(cubin_buf, ctypes.POINTER(ctypes.c_char))
        cubin = bytes(np.ctypeslib.as_array(cubin_ptr, shape=(size,)))
        self._cubin_cache[cc] = cubin
        self._linkerinfo_cache[cc] = linker.info_log

        return cubin

    def get_cufunc(self):
        if self._entry_name is None:
            msg = "Missing entry_name - are you trying to get the cufunc " \
                  "for a device function?"
            raise RuntimeError(msg)

        ctx = devices.get_context()
        device = ctx.device

        cufunc = self._cufunc_cache.get(device.id, None)
        if cufunc:
            return cufunc

        cubin = self.get_cubin(cc=device.compute_capability)
        module = ctx.create_module_image(cubin)

        # Load
        cufunc = module.get_function(self._entry_name)

        # Populate caches
        self._cufunc_cache[device.id] = cufunc

        return cufunc

    def get_linkerinfo(self, cc):
        try:
            return self._linkerinfo_cache[cc]
        except KeyError:
            raise KeyError(f'No linkerinfo for CC {cc}')

    def get_sass(self, cc=None):
        return disassemble_cubin(self.get_cubin(cc=cc))

    def add_ir_module(self, mod):
        self._raise_if_finalized()
        if self._module is not None:
            raise RuntimeError('CUDACodeLibrary only supports one module')
        self._module = mod

    def add_linking_library(self, library):
        library._ensure_finalized()

        # We don't want to allow linking more libraries in after finalization
        # because our linked libraries are modified by the finalization, and we
        # won't be able to finalize again after adding new ones
        self._raise_if_finalized()

        self._linking_libraries.add(library)

    def add_linking_file(self, filepath):
        self._linking_files.add(filepath)

    def get_function(self, name):
        for fn in self._module.functions:
            if fn.name == name:
                return fn
        raise KeyError(f'Function {name} not found')

    @property
    def modules(self):
        return [self._module] + [mod for lib in self._linking_libraries
                                 for mod in lib.modules]

    def finalize(self):
        # Unlike the CPUCodeLibrary, we don't invoke the binding layer here -
        # we only adjust the linkage of functions. Global kernels (with
        # external linkage) have their linkage untouched. Device functions are
        # set linkonce_odr to prevent them appearing in the PTX.

        self._raise_if_finalized()

        # Note in-place modification of the linkage of functions in linked
        # libraries. This presently causes no issues as only device functions
        # are shared across code libraries, so they would always need their
        # linkage set to linkonce_odr. If in a future scenario some code
        # libraries require linkonce_odr linkage of functions in linked
        # modules, and another code library requires another linkage, each code
        # library will need to take its own private copy of its linked modules.
        #
        # See also discussion on PR #890:
        # https://github.com/numba/numba/pull/890
        for library in self._linking_libraries:
            for fn in library._module.functions:
                if not fn.is_declaration:
                    fn.linkage = 'linkonce_odr'

        self._finalized = True

    def _reduce_states(self):
        """
        Reduce the instance for serialization. We retain the PTX and cubins,
        but loaded functions are discarded. They are recreated when needed
        after deserialization.
        """
        if self._linking_files:
            msg = ('cannot pickle CUDACodeLibrary function with additional '
                   'libraries to link against')
            raise RuntimeError(msg)
        return dict(
            codegen=self._codegen,
            name=self.name,
            entry_name=self._entry_name,
            module=self._module,
            linking_libraries=self._linking_libraries,
            ptx_cache=self._ptx_cache,
            cubin_cache=self._cubin_cache,
            linkerinfo_cache=self._linkerinfo_cache,
            max_registers=self._max_registers,
            nvvm_options=self._nvvm_options
        )

    @classmethod
    def _rebuild(cls, codegen, name, entry_name, module, linking_libraries,
                 ptx_cache, cubin_cache, linkerinfo_cache, max_registers,
                 nvvm_options):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        super(cls, instance).__init__(codegen, name)
        instance._entry_name = entry_name

        instance._module = module
        instance._linking_libraries = linking_libraries
        instance._linking_files = set()

        instance._ptx_cache = ptx_cache
        instance._cubin_cache = cubin_cache
        instance._linkerinfo_cache = linkerinfo_cache
        instance._cufunc_cache = {}

        instance._max_registers = max_registers
        instance._nvvm_options = nvvm_options


class JITCUDACodegen(Codegen):
    """
    This codegen implementation for CUDA only generates optimized LLVM IR.
    Generation of PTX code is done separately (see numba.cuda.compiler).
    """

    _library_class = CUDACodeLibrary

    def __init__(self, module_name):
        self._data_layout = nvvm.default_data_layout
        self._target_data = ll.create_target_data(self._data_layout)

    def _create_empty_module(self, name):
        ir_module = ir.Module(name)
        ir_module.triple = CUDA_TRIPLE
        if self._data_layout:
            ir_module.data_layout = self._data_layout
        nvvm.add_ir_version(ir_module)
        return ir_module

    def _add_module(self, module):
        pass
