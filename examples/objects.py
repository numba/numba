#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba import jit

class MyClass(object):
    def mymethod(self, arg):
        return arg * 2

@jit
def call_method(obj):
    print(obj.mymethod("hello"))  # object result
    mydouble = obj.mymethod(10.2) # native double
    print(mydouble * 2)           # native multiplication
    
call_method(MyClass())
