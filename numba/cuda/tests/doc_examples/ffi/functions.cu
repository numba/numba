// Foreign function example: multiplication of a pair of floats

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

