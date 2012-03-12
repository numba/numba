Primality Test for ufunc.

 * Implement the while-loop

 * Use LLVM optimization passes?

    import math

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        i = 2
        halfnum = int(math.ceil(math.sqrt(n)))
        while i <= halfnum:
            if n % i == 0:
                return False
            i += 1
        return True
