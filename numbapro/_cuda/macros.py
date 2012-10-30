# cuda.grid(1) -> threadIdx.x + blockIdx.x * blockDim.x
# cuda.grid(2) -> cuda.grid(1); threadIdx.y + blockIdx.y * blockDim.y
# cuda.grid(3) -> cuda.grid(2); threadIdx.z + blockIdx.z * blockDim.z

class grid:
    pass

