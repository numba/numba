from os.path import dirname, join
from unittest.case import TestCase
from unittest.suite import TestSuite
from subprocess import STDOUT, check_output, CalledProcessError
from numba.testing.ddt import ddt, data
from numba.testing.notebook import NotebookTest
from numba import cuda

test_scripts = [
    'binarytree.py',
    'bubblesort.py',
    'cffi_example.py',
    'compile_with_pycc.py',
    'ctypes_example.py',
    #'cuda_mpi.py',
    'fbcorr.py',
    'jitclass.py',
    'linkedlist.py',
    'movemean.py',
    'nogil.py',
    'objects.py',
    'ra24.py',
    'structures.py',
    'sum.py',
    'ufuncs.py',
    'blackscholes/blackscholes.py',
    'blackscholes/blackscholes_numba.py',
    'laplace2d/laplace2d.py',
    'laplace2d/laplace2d-numba.py',
    'example.py',
    'mandel.py',
    'mandel/mandel_vectorize.py',
    'mandel/mandel_autojit.py',
    'nbody/nbody.py',
    'nbody/nbody_modified_by_MarkHarris.py',
    'vectorize/sum.py',
    'vectorize/polynomial.py',
]

if cuda.is_available():
    test_scripts.extend([
    'blackscholes/blackscholes_cuda.py',
    'cudajit/matmul.py',
    'cudajit/matmul_smem.py',
    'cudajit/sum.py',
    'laplace2d/laplace2d-numba-cuda.py',
    'laplace2d/laplace2d-numba-cuda-improve.py',
    'laplace2d/laplace2d-numba-cuda-smem.py',
    'vectorize/cuda_polynomial.py',
    ])

notebooks = ['j0 in Numba.ipynb', # contains errors
             'LinearRegr.ipynb',
             'numba.ipynb',
             'Using Numba.ipynb']


@ddt
class TestExample(TestCase):
    """Test adapter to validate example applets."""

    @data(*test_scripts)
    def test(self, script):
        script = join(dirname(dirname(__file__)), script)
        status = 0
        try:
            check_output(script, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            status = e.returncode
            print(e.output.decode())
        self.assertEqual(status, 0)


@ddt
class NBTest(NotebookTest):

    @data(*notebooks)
    def test(self, nb):
        test = 'check_error'  # This is the only currently supported test type
        notebook = join(dirname(dirname(__file__)), 'notebooks', nb)
        self._test_notebook(notebook, test)


def load_tests(loader, tests, pattern):

    notebooks = loader.loadTestsFromTestCase(NBTest)
    examples = loader.loadTestsFromTestCase(TestExample)
    return TestSuite([notebooks, examples])

