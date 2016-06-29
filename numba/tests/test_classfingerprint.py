from __future__ import print_function, division, absolute_import

import pickle
import copy
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
    "a docstring"
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

    def test_considered_members_parent(self):
        cfp = ClassFingerPrint(Parent)
        ccm = cfp.considered_class_members()
        expected_members = ['__doc__', '__module__', '__weakref__',
                            '__dict__', 'a_classmethod', 'a_staticmethod',
                            'foo', 'prop']
        self.assertEqual(set(ccm.keys()), set(expected_members))
        # check member types
        self.assertEqual(ccm['a_classmethod'][0], 'classmethod')
        self.assertIsInstance(ccm['a_classmethod'][1], bytes)

        self.assertEqual(ccm['a_staticmethod'][0], 'staticmethod')
        self.assertIsInstance(ccm['a_staticmethod'][1], bytes)

        self.assertEqual(ccm['foo'][0], 'function')
        self.assertIsInstance(ccm['foo'][1], bytes)

        self.assertEqual(ccm['prop'][0], 'property')
        fget, fset, fdel = ccm['prop'][1]
        self.assertIsInstance(fget, bytes)
        self.assertIsInstance(fset, bytes)
        self.assertIsNone(fdel)

        self.assertEqual(ccm['__weakref__'][0], 'getset_descriptor')
        self.assertEqual(ccm['__doc__'], None)
        self.assertEqual(ccm['__module__'], __name__)

    def test_considered_members_child(self):
        cfp = ClassFingerPrint(Child)
        ccm = cfp.considered_class_members()
        expected_members = ['__doc__', '__module__', 'bar']
        self.assertEqual(set(ccm.keys()), set(expected_members))
        self.assertEqual(ccm['bar'][0], 'function')
        self.assertIsInstance(ccm['bar'][1], bytes)
        self.assertEqual(ccm['__doc__'], "a docstring")
        self.assertEqual(ccm['__module__'], __name__)

    def test_deep_copy(self):
        # parent
        parent_cloned = copy.deepcopy(Parent)
        self.assertEqual(ClassFingerPrint(Parent),
                         ClassFingerPrint(parent_cloned))
        # child
        child_cloned = copy.deepcopy(Child)
        self.assertEqual(ClassFingerPrint(Child),
                         ClassFingerPrint(child_cloned))

    def test_shallow_copy(self):
        # parent
        parent_cloned = copy.copy(Parent)
        self.assertEqual(ClassFingerPrint(Parent),
                         ClassFingerPrint(parent_cloned))
        # child
        child_cloned = copy.copy(Child)
        self.assertEqual(ClassFingerPrint(Child),
                         ClassFingerPrint(child_cloned))

    def test_pickled_class(self):
        # parent
        parent_cloned = pickle.loads(pickle.dumps(Parent))
        self.assertEqual(ClassFingerPrint(Parent),
                         ClassFingerPrint(parent_cloned))
        # child
        child_cloned = pickle.loads(pickle.dumps(Child))
        self.assertEqual(ClassFingerPrint(Child),
                         ClassFingerPrint(child_cloned))

    def test_patching_parent(self):
        # dynamically create a clone of Parent class
        parent_cloned = type('Parent', Parent.__bases__, Parent.__dict__.copy())
        child_cloned = type('Child', (parent_cloned,), Child.__dict__.copy())
        # equal initially
        self.assertEqual(ClassFingerPrint(Parent).hexdigest(),
                         ClassFingerPrint(parent_cloned).hexdigest())
        self.assertEqual(ClassFingerPrint(Child).hexdigest(),
                         ClassFingerPrint(child_cloned).hexdigest())
        # monkey patch
        def newfoo(self):
            pass
        parent_cloned.foo = newfoo
        self.assertIs(_unwrap_method(parent_cloned.foo), newfoo)
        self.assertIsNot(_unwrap_method(parent_cloned.foo),
                         _unwrap_method(Parent.foo))
        # not equal after patching
        self.assertNotEqual(ClassFingerPrint(Parent).hexdigest(),
                            ClassFingerPrint(parent_cloned).hexdigest())
        self.assertNotEqual(ClassFingerPrint(Child).hexdigest(),
                            ClassFingerPrint(child_cloned).hexdigest())

    def test_patching_child(self):
        # dynamically create a clone of Child class
        parent_cloned = type('Parent', Parent.__bases__, Parent.__dict__.copy())
        child_cloned = type('Child', (parent_cloned,), Child.__dict__.copy())
        # equal initially
        self.assertEqual(ClassFingerPrint(Parent).hexdigest(),
                         ClassFingerPrint(parent_cloned).hexdigest())
        self.assertEqual(ClassFingerPrint(Child).hexdigest(),
                         ClassFingerPrint(child_cloned).hexdigest())
        # monkey patch
        def newbar(self):
            pass
        child_cloned.bar = newbar
        self.assertIs(_unwrap_method(child_cloned.bar), newbar)
        self.assertIsNot(_unwrap_method(child_cloned.bar),
                         _unwrap_method(Child.bar))
        # not equal after patching
        self.assertNotEqual(ClassFingerPrint(Child).hexdigest(),
                            ClassFingerPrint(child_cloned).hexdigest())
        # parent is still equal
        self.assertEqual(ClassFingerPrint(Parent).hexdigest(),
                         ClassFingerPrint(parent_cloned).hexdigest())


def _unwrap_method(method):
    # for python2 unbound method
    return getattr(method, '__func__', method)


if __name__ == '__main__':
    unittest.main()
