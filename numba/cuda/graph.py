from ctypes import c_void_p, addressof, c_uint, POINTER, Structure
from numba.cuda.errors import normalize_kernel_dimensions
from numba.cuda.cudadrv.driver import driver, is_device_memory, device_ctypes_pointer, byref
from numba.cuda.compiler import AutoJitCUDAKernel, CUDAKernel


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
        return builded


class EmptyNode(Node):
    def build(self, graph=None):
        builded = super().build(graph)
        driver.cuGraphAddEmptyNode(byref(builded.handle), builded.graph.handle,
                                   builded.deps, len(builded.deps))
        return builded


class (Node):
    def build(self, graph=None):
        builded = super().build(graph)
        driver.cuGraphAddEmptyNode(byref(builded.handle), builded.graph.handle,
                                   builded.deps, len(builded.deps))
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


class Graph:
    def __init__(self, flags=0):
        self.handle = c_void_p()
        driver.cuGraphCreate(byref(self.handle), flags)

    def instantiate(self):
        return GraphExec(self)


class GraphExec:
    def __init__(self, graph):
        self.handle = c_void_p()
        driver.cuGraphInstantiate(byref(self.handle), graph.handle, c_void_p(), c_void_p(), 0)

    def launch(self, stream=None):
        driver.cuGraphLaunch(self.handle, stream.handle if stream else 0)
