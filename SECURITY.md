# Numba Security Policy

## Reporting Security Issues

If you believe you have found a security vulnerability in Numba, please open a security advisory here: https://github.com/numba/numba/security/advisories/new

We will respond promptly to security issues, and if we do not believe it is a security issue, we may ask you to open a public issue instead.

## Numba Security Assumptions

Numba is a just-in-time compiler which can generate arbitrary LLVM IR, compile it, and load it immediately into the running process. As a result, the compiler _should not_ be used with untrusted inputs. This includes both the function being compiled, but also arguments to the compiler decorators (`@jit`, `@vectorize`, etc) that control code generation. Additionally, Numba-compiled code, like C and FORTRAN, is not intended to be memory-safe.
