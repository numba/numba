#include "cuda_fp16.h"

extern "C" __device__ int
hsin_wrapper(
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
hcos_wrapper(
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
hlog_wrapper(
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
hlog10_wrapper(
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
hlog2_wrapper(
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
hexp_wrapper(
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
hexp10_wrapper(
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
hexp2_wrapper(
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
hsqrt_wrapper(
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
hrsqrt_wrapper(
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
hfloor_wrapper(
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
hceil_wrapper(
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
hrcp_wrapper(
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
hrint_wrapper(
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
htrunc_wrapper(
  short* return_value,
  short x
)
{
  __half retval = htrunc(__short_as_half (x));

  *return_value = __half_as_short (retval);
  // Signal that no Python exception occurred
  return 0;
}

