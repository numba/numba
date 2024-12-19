import os
import shutil
import subprocess
import sys
import unittest

from numba.tests.support import temp_directory
import cloudpickle


class TestObjectMode(unittest.TestCase):
    # Tests object mode uses the correct globals when loading from cloudpickle:
    # https://github.com/numba/numba/issues/9786
    _numba_parallel_test_ = False

    source_text_file = """
import numba
import cloudpickle
import os

VALUE = 1

@numba.jit
def func():
    with numba.objmode(val = "int64"):
        val = VALUE
    return val

pf = cloudpickle.dumps(func)

pickle_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "pf.pkl")

with open(pickle_path, "wb") as cpf:
    cpf.write(pf)
"""

    def setUp(self):
        self.tempdir = temp_directory('test_objmode_globals_cloudpickle_loc')

        self.pickle_func_py = os.path.join(self.tempdir, 'pickle_func.py')
        with open(self.pickle_func_py, 'w') as fout:
            print(self.source_text_file, file=fout)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_object_mode_globals_cloudpickle(self):
        # Execute pickle_func.py
        popen = subprocess.Popen([sys.executable, self.pickle_func_py],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        msg = f"stdout:\n{out.decode()}\n\nstderr:\n{err.decode()}"
        self.assertEqual(popen.returncode, 0, msg=msg)

        # load pickled function and try and run
        picked_func_path = os.path.join(self.tempdir, "pf.pkl")
        with open(picked_func_path, "rb") as cpf:
            pf = cpf.read()

        func = cloudpickle.loads(pf)

        val = func()

        self.assertEqual(val, 1)
