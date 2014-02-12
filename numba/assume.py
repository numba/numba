"""
A place to store all assumptions made in various part of the code base.

This allow us to do a usage analysis to discover all code that is assuming
something.

Each assumption is defined as a global variable.  Its value is the
description of the assumption.  Code that makes the assumption should
`assert the_assumption`
"""

return_argument_array_only = '''Only array passed into the function as
argument can be returned'''


