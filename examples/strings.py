# -*- coding: utf-8 -*-

"""
Example of using strings with numba using libc and some basic string
functionality.
"""

from __future__ import division, absolute_import
import struct
import socket

import numba as nb
import cffi

ffi = cffi.FFI()
ffi.cdef("""
void abort(void);
char *strstr(const char *s1, const char *s2);
int atoi(const char *str);
char *strtok(char *restrict str, const char *restrict sep);
""")

lib = ffi.dlopen(None)

# For now, we need to make these globals so numba will recognize them
abort, strstr, atoi, strtok = lib.abort, lib.strstr, lib.atoi, lib.strtok

int8_p = nb.int8.pointer()
int_p  = nb.int_.pointer()

@nb.autojit(nopython=True)
def parse_int_strtok(s):
    """
    Convert an IP address given as a string to an int, similar to
    socket.inet_aton(). Performs no error checking!
    """
    result = nb.uint32(0)
    current = strtok(s, ".")
    for i in range(4):
        byte = atoi(current)
        shift = (3 - i) * 8
        result |= byte << shift
        current = strtok(int_p(nb.NULL), ".")

    return result

@nb.autojit(nopython=True)
def parse_int_manual(s):
    """
    Convert an IP address given as a string to an int, similar to
    socket.inet_aton(). Performs no error checking!
    """
    result = nb.uint32(0)
    end = len(s)
    start = 0
    shift = 3
    for i in range(end):
        if s[i] == '.'[0] or i == end - 1:
            byte = atoi(int8_p(s) + start)
            result |= byte << (shift * 8)
            shift -= 1
            start = i + 1

    return result

result1 = parse_int_strtok('192.168.1.2')
result2 = parse_int_manual('1.2.3.4')
print(socket.inet_ntoa(struct.pack('>I', result1)))
print(socket.inet_ntoa(struct.pack('>I', result2)))