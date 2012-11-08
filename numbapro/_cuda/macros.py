# cuda.grid(1) -> threadIdx.x + blockIdx.x * blockDim.x
# cuda.grid(2) -> cuda.grid(1); threadIdx.y + blockIdx.y * blockDim.y
# cuda.grid(3) is impossible as we don't have blockIdx.z (block is 2D)

class grid:
    pass

