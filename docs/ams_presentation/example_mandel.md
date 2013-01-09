# Example: Mandelbrot

```python
    @autojit
    def mandel(x, y, max_iters):
        i = 0
        c = complex(x,y)
        z = 0.0j
        for i in range(max_iters):
            z = z ** 2 + c
            if (z.real ** 2 + z.imag ** 2) >= 4:
                return i

        return 255
```

# Example Mandelbrot

```python
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
```

# Example Mandelbrot

## 1000x speedup !!!

![Mandelbrot](mandel.jpg)
