"""
    Type System utilities common across type systems.
"""


def parse_integer_bitwidth(name):
    bitwidth = int(name.split('int')[-1])
    return bitwidth


def parse_integer_signed(name):
    signed = 'int' in name and 'uint' not in name
    return signed
