from os.path import dirname, join
from unittest.case import TestCase
from unittest.suite import TestSuite
from subprocess import STDOUT, call, check_output, CalledProcessError
from numba.tests.ddt import ddt, data

test_scripts = [
    'bubblesort.py',
    'cffi_example.py',
    'compile_with_pycc.py',
    'ctypes_example.py',
    #'cuda_mpi.py',
    'fbcorr.py',
    'movemean.py',
    'nogil.py',
    'objects.py',
    #   Error: assertion error
    #'pycc_example.py',
    'ra24.py',
    'structures.py',
    'sum.py',
    'ufuncs.py',
    'blackscholes/blackscholes.py',
    'blackscholes/blackscholes_numba.py',
    'blackscholes/blackscholes_cuda.py',
    'cudajit/matmul.py',
    'cudajit/matmul_smem.py',
    'cudajit/sum.py',
    #   This one runs forever...
    #'laplace2d/laplace2d.py',
    #   KeyError: "Does not support option: 'backend'"
    #'laplace2d/laplace2d-numba.py',
    #   ctypes.ArgumentError: argument 2: <class 'TypeError'>: wrong type
    #'laplace2d/laplace2d-numba-cuda.py',
    #'laplace2d/laplace2d-numba-cuda-improve.py',
    #   Error: ArgumentError exception
    #'laplace2d/laplace2d-numba-cuda-smem.py',
    #   The following scripts are interactive
    #'example.py',
    #'mandel.py',
    #'mandel/mandel_vectorize.py',
    #'mandel/mandel_autojit.py',
    'nbody/nbody.py',
    'nbody/nbody_modified_by_MarkHarris.py',
    # Missing input files !?
    #'vectorize/sum.py',
    'vectorize/polynomial.py',
    'vectorize/cuda_polynomial.py',
]

@ddt
class TestExample(TestCase):
    """Test adapter to validate example applets."""

    @data(*test_scripts)
    def test(self, script):
        script=join(dirname(dirname(__file__)), script)
        status = 0
        try:
            out = check_output(script, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            status = e.returncode
            print(e.output)
        self.assertEqual(status, 0)


def load_tests(loader, tests, pattern):

    return loader.loadTestsFromTestCase(TestExample)

