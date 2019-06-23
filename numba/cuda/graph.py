from ctypes import c_void_p, addressof, c_uint, c_size_t, POINTER, Structure, byref
from numba.cuda import current_context
from numba.cuda.errors import normalize_kernel_dimensions
from numba.cuda.compiler import AutoJitCUDAKernel, CUDAKernel
from numba.cuda.cudadrv.enums import CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST
from numba.cuda.cudadrv.drvapi import cu_device_ptr, cu_array
from numba.cuda.cudadrv.driver import driver, is_device_memory, device_ctypes_pointer, \
    host_pointer, device_pointer


class KernelParams(Structure):
    _fields_ = [
        ('func', c_void_p),
        ('gridDimX', c_uint),
        ('gridDimY', c_uint),
        ('gridDimZ', c_uint),
        ('blockDimX', c_uint),
        ('blockDimY', c_uint),
        ('blockDimZ', c_uint),
        ('sharedMemBytes', c_uint),
        ('kernelParams', POINTER(c_void_p)),
        ('extra', POINTER(c_void_p)),
    ]


class MemcpyParams(Structure):
    _fields_ = [
        ('srcXInBytes', c_size_t),
        ('srcY', c_size_t),
        ('srcZ', c_size_t),
        ('srcLOD', c_size_t),
        ('srcMemoryType', c_uint),
        ('srcHost', c_void_p),
        ('srcDevice', cu_device_ptr),
        ('srcArray', cu_array),
        ('reserved0', c_void_p),
        ('srcPitch', c_size_t),
        ('srcHeight', c_size_t),

        ('dstXInBytes', c_size_t),
        ('dstY', c_size_t),
        ('dstZ', c_size_t),
        ('dstLOD', c_size_t),
        ('dstMemoryType', c_uint),
        ('dstHost', c_void_p),
        ('dstDevice', cu_device_ptr),
        ('dstArray', cu_array),
        ('reserved1', c_void_p),
        ('dstPitch', c_size_t),
        ('dstHeight', c_size_t),

        ('WidthInBytes', c_size_t),
        ('Height', c_size_t),
        ('Depth', c_size_t),
    ]


class MemsetParams(Structure):
    _fields_ = [
        ('dst', cu_device_ptr),
        ('pitch', c_size_t),
        ('value', c_uint),
        ('elementSize', c_uint),
        ('width', c_size_t),
        ('height', c_size_t),
    ]


class Node:
    def __init__(self, deps=None):
        self.deps = deps or []
        self.builded = { }

    def build(self, graph=None):
        if not graph:
            graph = Graph()

        if graph in self.builded:
            return self.builded[graph]

        args = [dep.build(graph).handle for dep in self.deps]
        deps = (c_void_p * len(self.deps))(*args)
        builded = self.builded[graph] = BuildedNode(c_void_p(), graph, deps)
        graph.nodes.append(self)
        return builded


class EmptyNode(Node):
    def build(self, graph=None):
        builded = super().build(graph)
        driver.cuGraphAddEmptyNode(byref(builded.handle), builded.graph.handle,
                                   builded.deps, len(builded.deps))
        return builded


class MemcpyNode(Node):
    def __init__(self, deps=None, params=None, ctx=None):
        self.params = params or { }
        self.ctx = ctx
        super().__init__(deps)

    def _get_params(self):
        params = MemcpyParams()

        params.srcXInBytes = self.params.get('srcXInBytes', 0)
        params.srcY = self.params.get('srcY', 0)
        params.srcZ = self.params.get('srcZ', 0)
        params.srcLOD = self.params.get('srcLOD', 0)
        params.srcMemoryType = self.params.get('srcMemoryType')
        if not params.srcMemoryType:
            raise Exception('params.srcMemoryType should be set')
        srcHost = self.params.get('srcHost')
        params.srcHost = None if srcHost is None else host_pointer(srcHost)
        srcDevice = self.params.get('srcDevice')
        params.srcDevice = device_pointer(srcDevice) if srcDevice else 0
        params.srcArray = None # TODO
        params.reserved0 = None
        params.srcPitch = self.params.get('srcPitch', 0)
        params.srcHeight = self.params.get('srcHeight', 0)

        params.dstXInBytes = self.params.get('dstXInBytes', 0)
        params.dstY = self.params.get('dstY', 0)
        params.dstZ = self.params.get('dstZ', 0)
        params.dstLOD = self.params.get('dstLOD', 0)
        params.dstMemoryType = self.params.get('dstMemoryType')
        if not params.dstMemoryType:
            raise Exception('params.dstMemoryType should be set')
        dstHost = self.params.get('dstHost')
        params.dstHost = None if dstHost is None else host_pointer(dstHost)
        dstDevice = self.params.get('dstDevice')
        params.dstDevice = device_pointer(dstDevice) if dstDevice else 0
        params.dstArray = None # TODO
        params.reserved1 = None
        params.dstPitch = self.params.get('dstPitch', 0)
        params.dstHeight = self.params.get('dstHeight', 0)

        params.WidthInBytes = self.params.get('WidthInBytes', 0)
        if not params.WidthInBytes >= 0:
            raise Exception('params.WidthInBytes should be the byte length of array')
        params.Height = self.params.get('Height', 1)
        params.Depth = self.params.get('Depth', 1)
        return params

    def build(self, graph=None):
        builded = super().build(graph)
        params = self._get_params()
        ctx = self.ctx or current_context()
        driver.cuGraphAddMemcpyNode(byref(builded.handle), builded.graph.handle,
                                    builded.deps, len(builded.deps), byref(params), ctx.handle)
        return builded


class MemcpyHtoDNode(MemcpyNode):
    def __init__(self, dst, src, size, deps=None, params=None, ctx=None):
        def_pars = {
            'srcHost': src, 'srcMemoryType': CU_MEMORYTYPE_HOST,
            'dstDevice': dst, 'dstMemoryType': CU_MEMORYTYPE_DEVICE,
            'WidthInBytes': size,
        }
        if params:
            def_pars.update(params)
        super().__init__(deps, def_pars, ctx)


class MemcpyDtoHNode(MemcpyNode):
    def __init__(self, dst, src, size, deps=None, params=None, ctx=None):
        def_pars = {
            'srcDevice': src, 'srcMemoryType': CU_MEMORYTYPE_DEVICE,
            'dstHost': dst, 'dstMemoryType': CU_MEMORYTYPE_HOST,
            'WidthInBytes': size,
        }
        if params:
            def_pars.update(params)
        super().__init__(deps, def_pars, ctx)


class MemsetNode(Node):
    def __init__(self, arr, size, val, deps=None, params=None, ctx=None):
        self.arr = arr
        self.size = size
        self.val = val
        self.params = params or { }
        self.ctx = ctx
        super().__init__(deps)

    def _get_params(self):
        params = MemsetParams()
        params.dst = device_pointer(self.arr)
        params.width = self.size
        params.value = self.val
        params.elementSize = self.params.get('elementSize', 4)
        if params.elementSize not in [1, 2, 4]:
            raise Exception('only 1, 2 or 4 is allowed for elementSize')
        params.height = self.params.get('height', 1)
        params.pitch = self.params.get('pitch', 1)
        return params

    def build(self, graph=None):
        builded = super().build(graph)
        params = self._get_params()
        ctx = self.ctx or current_context()
        driver.cuGraphAddMemsetNode(byref(builded.handle), builded.graph.handle,
                                    builded.deps, len(builded.deps), byref(params), ctx.handle)
        return builded


class KernelNode(Node):
    def __init__(self, kernel, args=None, deps=None, params=None):
        self.kernel = kernel
        self.args = args or []
        self.params = params or { }
        super().__init__(deps)

    def _get_params(self):
        params = KernelParams()

        gridDim, blockDim = self.params.get('gridDim', (1, 1, 1)), self.params.get('blockDim', (1, 1, 1))
        gridDim, blockDim = normalize_kernel_dimensions(gridDim, blockDim)
        params.gridDimX = self.params.get('gridDimX', gridDim[0])
        params.gridDimY = self.params.get('gridDimY', gridDim[1])
        params.gridDimZ = self.params.get('gridDimZ', gridDim[2])
        params.blockDimX = self.params.get('blockDimX', blockDim[0])
        params.blockDimY = self.params.get('blockDimY', blockDim[1])
        params.blockDimZ = self.params.get('blockDimZ', blockDim[2])
        params.sharedMemBytes = self.params.get('sharedMemBytes', 0)

        if isinstance(self.kernel, AutoJitCUDAKernel):
            kernel = self.kernel.specialize(*self.args)
        elif isinstance(self.kernel, CUDAKernel):
            kernel = self.kernel
        else:
            raise Exception('invalid kernel type "%s"' % type(self.kernel).__name__)
        params.func = kernel._func.get().handle

        retr, kernel_args = [], []
        for t, v in zip(kernel.argument_types, self.args):
            kernel._prepare_args(t, v, 0, retr, kernel_args)

        # TODO: take care of retr after graph launched
        if len(retr):
            raise Exception('host array as kernel node args not supported yet')

        param_vals = []
        for arg in kernel_args:
            if is_device_memory(arg):
                param_vals.append(addressof(device_ctypes_pointer(arg)))
            else:
                param_vals.append(addressof(arg))

        params.kernelParams = (c_void_p * len(param_vals))(*param_vals) if len(param_vals) else None
        params.extra = self.params.get('extra', None)
        return params

    def build(self, graph=None):
        builded = super().build(graph)
        params = self._get_params()
        driver.cuGraphAddKernelNode(byref(builded.handle), builded.graph.handle,
                                    builded.deps, len(builded.deps), byref(params))
        return builded


class BuildedNode:
    def __init__(self, handle, graph, deps):
        self.handle = handle
        self.graph = graph
        self.deps = deps

    def instantiate(self):
        return GraphExec(self.graph)

    def launch(self, stream=None):
        return GraphExec(self.graph).launch(stream)

    def destroy(self):
        self.graph.destroy()


class Graph:
    def __init__(self, flags=0):
        self.handle = c_void_p()
        self.nodes = []
        driver.cuGraphCreate(byref(self.handle), flags)

    def destroy(self):
        for node in self.nodes:
            del node.builded[self]
        if self.handle is not None:
            driver.cuGraphDestroy(self.handle)
        self.handle = None

    def instantiate(self):
        if self.handle is None:
            raise Exception('graph already destroyed!')
        return GraphExec(self)


class GraphExec:
    def __init__(self, graph):
        self.handle = c_void_p()
        driver.cuGraphInstantiate(byref(self.handle), graph.handle, c_void_p(), c_void_p(), 0)

    def launch(self, stream=None):
        driver.cuGraphLaunch(self.handle, stream.handle if stream else 0)
