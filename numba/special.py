"""
Special compiler-recognized numba functions and attributes.
"""

__all__ = ['NULL']

class NumbaDotNULL(object):
    "NULL pointer"

NULL = NumbaDotNULL()