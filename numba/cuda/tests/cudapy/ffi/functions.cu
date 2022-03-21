// Foreign function example: multiplication of a pair of floats
#include "cuda_fp16.h"

extern "C" __device__ int
mul_f32_f32(
  float* return_value,
  float x,
  float y)
{
  // Compute result and store in caller-provided slot
  *return_value = x * y;

  // Signal that no Python exception occurred
  return 0;
}



extern "C" __device__ int
hsin_wrapper(
  short* return_value,
  short x
)
{
  *return_value = hsin(x);

  // Signal that no Python exception occurred
  return 0;
}

extern "C" __device__ int
hcos_wrapper(
  short* return_value,
  short x
)
{
  *return_value = hcos(x);

  // Signal that no Python exception occurred
  return 0;
}