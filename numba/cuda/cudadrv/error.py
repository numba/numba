class CudaDriverError(Exception):
    pass


class CudaRuntimeError(Exception):
    pass


class CudaSupportError(ImportError):
    pass


class NvvmError(Exception):
    def __str__(self):
        return '\n'.join(map(str, self.args))


class NvvmSupportError(ImportError):
    pass


class NvvmWarning(Warning):
    pass
