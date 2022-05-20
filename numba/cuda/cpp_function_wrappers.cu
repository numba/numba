#include "cuda_fp16.h"

#define FNDEF(fname) __numba_wrapper_ ## fname

extern "C" __device__ int
FNDEF(hdiv)(
  short* return_value,
  short x,
  short y
)
{
  __half retval = __hdiv(__short_as_half (x), __short_as_half (y));
  
  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hsin)(
  short* return_value,
  short x
)
{
  __half retval = hsin(__short_as_half (x));
  
  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hcos)(
  short* return_value,
  short x
)
{
  __half retval = hcos(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hlog)(
  short* return_value,
  short x
)
{
  __half retval = hlog(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hlog10)(
  short* return_value,
  short x
)
{
  __half retval = hlog10(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hlog2)(
  short* return_value,
  short x
)
{
  __half retval = hlog2(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hexp)(
  short* return_value,
  short x
)
{
  __half retval = hexp(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hexp10)(
  short* return_value,
  short x
)
{
  __half retval = hexp10(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hexp2)(
  short* return_value,
  short x
)
{
  __half retval = hexp2(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hsqrt)(
  short* return_value,
  short x
)
{
  __half retval = hsqrt(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hrsqrt)(
  short* return_value,
  short x
)
{
  __half retval = hrsqrt(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hfloor)(
  short* return_value,
  short x
)
{
  __half retval = hfloor(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hceil)(
  short* return_value,
  short x
)
{
  __half retval = hceil(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hrcp)(
  short* return_value,
  short x
)
{
  __half retval = hrcp(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(hrint)(
  short* return_value,
  short x
)
{
  __half retval = hrint(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
FNDEF(htrunc)(
  short* return_value,
  short x
)
{
  __half retval = htrunc(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

