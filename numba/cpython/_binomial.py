"""Shared BTPE binomial sampler, factored out of the legacy ``np.random`` path
(``numba.cpython.randomimpl``) and the ``Generator`` path
(``numba.np.random.distributions``).

The two paths differ only in (1) how a uniform double is drawn and (2) the
``case == 52`` squeeze test (whose Stirling constants are frozen at NumPy's
legacy values for ``np.random`` but track NumPy's corrected values for
``Generator``). Both are injected, so the ~100 lines of rejection scaffolding
live here once. Keep in sync with NumPy's ``random_binomial_btpe``.
https://github.com/numpy/numpy/blob/2538439951286e1e8be2c66c9648bd58cea6e849/numpy/random/src/distributions/distributions.c#L622-L806
"""
import math

from numba.core.extending import register_jitable


def make_binomial_btpe(next_double, squeeze_reject):
    """Build a BTPE sampler closed over ``next_double(bitgen)`` (returns a
    uniform double; the legacy path ignores ``bitgen`` and draws
    ``np.random.random()``) and ``squeeze_reject(A, xm, m, n, y, r, q)`` (the
    ``case == 52`` acceptance test, ``True`` to reject). ``r`` is
    ``min(p, 1 - p)``; the caller owns the ``p > 0.5`` flip."""

    @register_jitable
    def binomial_btpe(bitgen, n, r):
        q = 1.0 - r
        fm = n * r + r
        m = int(math.floor(fm))
        p1 = math.floor(2.195 * math.sqrt(n * r * q) - 4.6 * q) + 0.5
        xm = m + 0.5
        xl = xm - p1
        xr = xm + p1
        c = 0.134 + 20.5 / (15.3 + m)
        a = (fm - xl) / (fm - xl * r)
        laml = a * (1.0 + a / 2.0)
        a = (xr - fm) / (xr * q)
        lamr = a * (1.0 + a / 2.0)
        p2 = p1 * (1.0 + 2.0 * c)
        p3 = p2 + c / laml
        p4 = p3 + c / lamr

        case = 10
        y = k = 0
        while 1:
            if case == 10:
                nrq = n * r * q
                u = next_double(bitgen) * p4
                v = next_double(bitgen)
                if u > p1:
                    case = 20
                    continue
                y = int(math.floor(xm - p1 * v + u))
                case = 60
                continue
            elif case == 20:
                if u > p2:
                    case = 30
                    continue
                x = xl + (u - p1) / c
                v = v * c + 1.0 - math.fabs(m - x + 0.5) / p1
                if v > 1.0:
                    case = 10
                    continue
                y = int(math.floor(x))
                case = 50
                continue
            elif case == 30:
                if u > p3:
                    case = 40
                    continue
                y = int(math.floor(xl + math.log(v) / laml))
                if (y < 0) or (v == 0.0):
                    case = 10
                    continue
                v = v * (u - p2) * laml
                case = 50
                continue
            elif case == 40:
                y = int(math.floor(xr - math.log(v) / lamr))
                if (y > n) or (v == 0.0):
                    case = 10
                    continue
                v = v * (u - p3) * lamr
                case = 50
                continue
            elif case == 50:
                k = abs(y - m)
                if (k > 20) and (k < (nrq / 2.0 - 1)):
                    case = 52
                    continue
                s = r / q
                a = s * (n + 1)
                F = 1.0
                if m < y:
                    for i in range(m + 1, y + 1):
                        F = F * (a / i - s)
                elif m > y:
                    for i in range(y + 1, m + 1):
                        F = F / (a / i - s)
                if v > F:
                    case = 10
                    continue
                case = 60
                continue
            elif case == 52:
                rho = (k / nrq) * ((k * (k / 3.0 + 0.625) +
                                    0.16666666666666666) / nrq + 0.5)
                t = -k * k / (2 * nrq)
                A = math.log(v)
                if A < (t - rho):
                    case = 60
                    continue
                if A > (t + rho):
                    case = 10
                    continue
                if squeeze_reject(A, xm, m, n, y, r, q):
                    case = 10
                    continue
                case = 60
                continue
            elif case == 60:
                return y

    return binomial_btpe
