from _declaration import Declaration, Configuration

seed = Configuration('seed', '''
seed [int]
    
    Seed for the PRNG.
''')

uniform = Declaration('uniform', '''
uniform(ary)

    Write random floating-point values into `ary`.
    The values are sampled from a uniform distribution.

    ary --- a contiguous array of float or double.

''')

normal  = Declaration('normal', '''
normal(ary, mean=0, sigma=1)

    Write random floating-point values into `ary`.
    The values are sampled from a normal distribution.

    ary   --- a contiguous array of float or double.
    mean  --- the center of the distribtion.
    sigma --- the std-deviation of the distribution.
''')
