import torch
import numba.cuda

def tensor_cuda_array_interface(tensor):
    if not tensor.device.type == "cuda":
        raise TypeError(
            "Can't convert cpu tensor to cuda array. "
            "Use Tensor.cuda() to copy the tensor to device memory first."
        )

    if tensor.requires_grad:
        raise RuntimeError(
            "Can't get cuda array interface for Variable that requires grad. "
            "Use var.detach() first."
        )

    if tensor.device.index != numba.cuda.get_current_device().id:
        raise ValueError(
            "tensor device: %r is not active numba context: %r" % (
                tensor.device, numba.cuda.current_contenxt()
            )
        )

    typestr = {
        torch.float16: "f2",
        torch.float32: "f4",
        torch.float64: "f8",
        torch.uint8: "u1",
        torch.int8: "i1",
        torch.int16: "i2",
        torch.int32: "i4",
        torch.int64: "i8",
    }[tensor.dtype]

    itemsize = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
    }[tensor.dtype]

    shape = tensor.shape
    strides = tuple(s * itemsize for s in tensor.stride())
    data = (tensor.data_ptr(), False)

    return dict(
        typestr=typestr,
        shape=shape,
        strides=strides,
        data=data,
        version=0,
    )

@numba.cuda.as_cuda_array.register(torch.Tensor)
def as_cuda_array(torch_tensor):
    return numba.cuda.from_cuda_array_interface(
        tensor_cuda_array_interface(torch_tensor),
        owner = torch_tensor
    )

@numba.cuda.is_cuda_array.register(torch.Tensor)
def is_cuda_array(torch_tensor):
    return torch_tensor.device.type == "cuda"

def test_array_adaptor():
    import pytest
    import math
    import numpy

    _torch_dtype_mapping = {
        float: torch.float32,
        bool: torch.uint8,
        numpy.float16: torch.float16,
        numpy.float32: torch.float32,
        numpy.float64: torch.float64,
        numpy.uint8: torch.uint8,
        numpy.bool_: torch.uint8,
        numpy.int8: torch.int8,
        numpy.int16: torch.int16,
        numpy.int32: torch.int32,
        numpy.int64: torch.int64,
    }

    ### Test converters for all torch data types.
    for dt in set(_torch_dtype_mapping.values()):
        if dt == torch.int8:
            # Skip tests of int8, not officially supported by pytorch
            continue

        cput = torch.arange(10).to(dt)
        npt = cput.numpy()

        ### cpu-tensors should not register as cuda arrays
        assert not is_cuda_array(cput)

        ### converting cpu-tensors should raise a TypeError
        ### mirroring Tensor.numpy() on cuda tensors
        with pytest.raises(TypeError):
            numba.cuda.as_cuda_array(cput)

        cudat = cput.to(device="cuda")

        ### cuda-tensors should register as cuda arrays
        assert is_cuda_array(cudat)

        ### cuda-tensors that require gradient should raise a RuntimeError
        ### mirror Tensor.numpy() on cpu tensors
        with pytest.raises(RuntimeError):
            numba.cuda.as_cuda_array(cudat.clone().requires_grad_(True))

        ### as_cuda_array returns DeviceNDArray of the shape type/stride/shape
        numba_view = numba.cuda.as_cuda_array(cudat)
        assert isinstance(numba_view, numba.cuda.devicearray.DeviceNDArray)
        assert numba_view.dtype == npt.dtype
        assert numba_view.strides == npt.strides
        assert numba_view.shape == cudat.shape
        # Pass back to cuda from host for fp16 comparisons
        assert (cudat == torch.tensor(numba_view.copy_to_host()
                                      ).to("cuda")).all()

        ### the DeviceNDArray is a view, modification propagates to the source
        cudat[:5] = math.pi
        # Pass back to cuda from host for fp16 comparisons
        assert (cudat == torch.tensor(numba_view.copy_to_host()
                                      ).to("cuda")).all()

        ### Can take views of strides slices of cuda arrays
        strided_cudat = cudat[::2]
        strided_numba_view = numba.cuda.as_cuda_array(strided_cudat)

        # Bug with copies of strided data device->host
        # would expect that we can copy back to device, but it raises error
        with pytest.raises((TypeError, ValueError)):
            assert (
                strided_cudat.to("cpu") == torch.tensor(
                    strided_numba_view.copy_to_host()
                )
            ).all()

        # instead generate a strided result buffer
        result_buffer = numpy.empty(10, dtype=strided_numba_view.dtype)
        result_view = result_buffer[::2]
        strided_numba_view.copy_to_host(result_view)
        # Pass back to cuda from host for fp16 comparisons
        assert (strided_cudat == torch.tensor(result_view).to("cuda")).all()
