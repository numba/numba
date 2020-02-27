import warnings
from numba import jit
from numba.core.errors import (NumbaDeprecationWarning,
                          NumbaPendingDeprecationWarning, NumbaWarning)
import unittest


class TestDeprecation(unittest.TestCase):

    def check_warning(self, warnings, expected_str, category):
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].category, category)
        self.assertIn(expected_str, str(warnings[0].message))
        self.assertIn("http://numba.pydata.org", str(warnings[0].message))

    def test_jitfallback(self):
        # tests that @jit falling back to object mode raises a
        # NumbaDeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore", category=NumbaWarning)
            warnings.simplefilter("always", category=NumbaDeprecationWarning)

            def foo():
                return []  # empty list cannot be typed
            jit(foo)()

            msg = ("Fall-back from the nopython compilation path to the object "
                   "mode compilation path")
            self.check_warning(w, msg, NumbaDeprecationWarning)

    def test_reflection_of_mutable_container(self):
        # tests that reflection in list/set warns
        def foo_list(a):
            return a.append(1)

        def foo_set(a):
            return a.add(1)

        for f in [foo_list, foo_set]:
            container = f.__name__.strip('foo_')
            inp = eval(container)([10, ])
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore", category=NumbaWarning)
                warnings.simplefilter("always",
                                      category=NumbaPendingDeprecationWarning)
                jit(nopython=True)(f)(inp)
                self.assertEqual(len(w), 1)
                self.assertEqual(w[0].category, NumbaPendingDeprecationWarning)
                warn_msg = str(w[0].message)
                msg = ("Encountered the use of a type that is scheduled for "
                       "deprecation")
                self.assertIn(msg, warn_msg)
                msg = ("\'reflected %s\' found for argument" % container)
                self.assertIn(msg, warn_msg)
                self.assertIn("http://numba.pydata.org", warn_msg)


if __name__ == '__main__':
    unittest.main()
