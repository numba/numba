# Copyright (c) 2012, Siu Kwan Lam
# All rights reserved.
#
# Contains most of the functions from math.h (libc).
#
# NOTE: ldexp, frexp and modf are not included yet. These functions take
#       a C-pointer. I have not figure not how to pointer for non-array types
#       yet.
#

from pymothoa.types import *

# trigonometric
sinf = 'sinf', Float, [Float]
sin  = 'sin', Double, [Double]

cosf = 'cosf', Float, [Float]
cos  = 'cos', Double, [Double]

tanf = 'tanf', Float, [Float]
tan  = 'tan', Double, [Double]

asinf = 'asinf', Float, [Float]
asin  = 'asin', Double, [Double]

acosf = 'acosf', Float, [Float]
acos  = 'acos', Double, [Double]

atanf = 'atanf', Float, [Float]
atan  = 'atan', Double, [Double]

# hyperbolic
sinhf = 'sinhf', Float, [Float]
sinh  = 'sinh', Double, [Double]

coshf = 'coshf', Float, [Float]
cosh  = 'cosh', Double, [Double]

tanhf = 'tanhf', Float, [Float]
tanh  = 'tanh', Double, [Double]

# power
sqrtf = 'sqrtf', Float, [Float]
sqrt  = 'sqrt', Double, [Double]

powf = 'powf', Float, [Float, Float]
pow  = 'pow', Double, [Double, Double]


# exponential
expf = 'expf', Float, [Float]
exp  = 'exp', Double, [Double]

logf = 'logf', Float, [Float]
log  = 'log', Double, [Double]

log10f = 'log10f', Float, [Float]
log10  = 'log10', Double, [Double]

# misc
fabsf = 'fabsf', Float, [Float]
fabs  = 'fabs', Double, [Double]

ceilf = 'ceilf', Float, [Float]
ceil  = 'ceil', Double, [Double]

floorf = 'floorf', Float, [Float]
floor  = 'floor', Double, [Double]

fmodf = 'fmodf', Float, [Float, Float]
fmod  = 'fmod', Double, [Double, Double]


