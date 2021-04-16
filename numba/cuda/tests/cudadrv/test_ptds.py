
def child_test():
    from numba import cuda, float32, void
    from time import perf_counter
    import numpy as np
    import threading

    N = 2 ** 16
    N_THREADS = 10
    N_ADDITIONS = 4096

    np.random.seed(1)
    x = np.random.random(N).astype(np.float32)
    r = np.zeros_like(x)

    xs = [cuda.to_device(x) for _ in range(N_THREADS)]
    rs = [cuda.to_device(r) for _ in range(N_THREADS)]

    n_threads = 256
    n_blocks = N // n_threads
    stream = cuda.default_stream()


    @cuda.jit(void(float32[::1], float32[::1]))
    def f(r, x):
        i = cuda.grid(1)

        if i > len(r):
            return

        # Accumulate x into r
        for j in range(N_ADDITIONS):
            r[i] += x[i]


    def kernel_thread(n):
        f[n_blocks, n_threads, stream](rs[n], xs[n])


    def main():
        print("Creating threads")
        threads = [threading.Thread(target=kernel_thread, args=(i,))
                   for i in range(N_THREADS)]

        print("Starting threads")
        start = perf_counter()

        for thread in threads:
            thread.start()

        print("Waiting for threads to finish")
        for thread in threads:
            thread.join()

        print("Synchronizing with device")
        cuda.synchronize()

        end = perf_counter()
        print(f"Elapsed time: {end - start}")

        print("Checking output")
        expected = x * N_ADDITIONS

        for i in range(N_THREADS):
            print(f"Checking output {i}")
            # Lower than usual tolerance because our method of accumulation is not
            # particularly accurate
            rtol = 1.0e-4
            np.testing.assert_allclose(rs[i].copy_to_host(), expected, rtol=rtol)

        print("Done!")


