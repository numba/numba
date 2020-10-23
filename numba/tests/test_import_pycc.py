from numba.pycc import CC
from numba import i8
from numba.core.types import DictType, unicode_type
from numba.typed import Dict
from numba.tests.support import TestCase
import unittest
import os
import glob


class TestImportStringHashing(TestCase):

    """
    Tests that numba compiled modules can be imported if they have
     functions that internally hash strings. See issue #6386.
    """
    def setUp(self):
        cc = CC('str_dict_module')

        @cc.export('in_str_dict',(DictType(unicode_type,i8),unicode_type))
        def in_str_dict(d,x):
            if(x not in d):
                d[x] = len(d)

        @cc.export('out_str_dict',DictType(unicode_type,i8)(unicode_type,))
        def out_str_dict(x):
            d = Dict.empty(unicode_type,i8)
            if(x not in d):
                d[x] = len(d)
            return d

        @cc.export('hash_str',(unicode_type,))
        def hash_str(x):
            return hash(x)

        cc.compile()

    def test_import_str_dicts(self):
        from numba.tests.str_dict_module import in_str_dict as _in_str_dict
        from numba.tests.str_dict_module import out_str_dict as _out_str_dict
        from numba.tests.str_dict_module import hash_str as _hash_str
        d = _out_str_dict('boop')
        _in_str_dict(d,'beep')
        self.assertEqual(d['beep'], 1)
        self.assertEqual(_hash_str('beep'), hash('beep'))

    def tearDown(self):
        os.remove(glob.glob("str_dict_module.*[dll|so]")[0])


if __name__ == "__main__":
    unittest.main()
