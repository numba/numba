from __future__ import print_function, division, absolute_import

from pprint import pprint

from numba.classfingerprint import ClassFingerPrint
from numba import unittest_support as unittest


class Parent(object):
    def foo(self):
        return True

    @classmethod
    def a_classmethod(self):
        pass

    @staticmethod
    def a_staticmethod(self):
        pass

    @property
    def prop(self):
        pass

    @prop.setter
    def prop(self, val):
        pass


class Child(Parent):
    def bar(self):
        return


class TestClassFingerprint(unittest.TestCase):
    def _check_digest(self, cls):
        cfp1 = ClassFingerPrint(cls)
        cfp2 = ClassFingerPrint(cls)
        self.assertEqual(cfp1.hexdigest(), cfp2.hexdigest())
        # str digest
        hexdigest = cfp1.hexdigest()
        self.assertIsInstance(hexdigest, str)
        self.assertEqual(len(hexdigest), 32)
        # bytes digest
        digest = cfp1.digest()
        self.assertIsInstance(digest, bytes)
        self.assertEqual(len(digest), 16)

    def test_digest_parent(self):
        self._check_digest(Parent)

    def test_digest_child(self):
        self._check_digest(Child)

    def test_digest_different(self):
        digest1 = ClassFingerPrint(Parent).hexdigest()
        digest2 = ClassFingerPrint(Child).hexdigest()
        self.assertNotEqual(digest1, digest2)



if __name__ == '__main__':
    unittest.main()
