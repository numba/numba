from vbench.benchmark import Benchmark
import datetime

common_setup = """
from numba import *
"""

setup = common_setup + """
@autojit
def mandel(x, y, max_iters):
    i = 0
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z**2 + c
        if abs(z)**2 >= 4:
            return i

    return 255

@autojit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image
"""

stmt = """
image = np.zeros((500, 750), dtype=np.uint8)
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
"""

bench = Benchmark(stmt, setup, name="mandelbrot",)
                  #start_date=datetime.datetime())