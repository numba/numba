from __future__ import print_function
import itertools
import numba.unittest_support as unittest
from numba.compiler import compile_isolated
from numba import types


def template(fromty, toty):
    def closure(self):
        def cast(x):
            y = x
            return y

        cres = compile_isolated(cast, args=[fromty], return_type=toty)
        self.assertAlmostEqual(cres.entry_point(1), 1)

    return closure


class TestNumberConversion(unittest.TestCase):
    """
    Test all int/float numeric conversion to ensure we have all the external
    dependencies to perform these conversions.
    """
    # NOTE: more implicit tests are in test_numberctor

    @classmethod
    def automatic_populate(cls):
        tys = types.integer_domain | types.real_domain
        for fromty, toty in itertools.permutations(tys, r=2):
            test_name = "test_{fromty}_to_{toty}".format(fromty=fromty,
                                                         toty=toty)
            setattr(cls, test_name, template(fromty, toty))


TestNumberConversion.automatic_populate()

if __name__ == '__main__':
    unittest.main()
