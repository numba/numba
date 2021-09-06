# -*- coding: utf-8 -*-
"""Test distance matrix calculations using CUDA/Numba."""
from scipy.spatial.distance import euclidean, cdist
from scipy.stats import wasserstein_distance as scipy_wasserstein_distance
from numpy.testing import assert_allclose
import numpy as np
import os
from importlib import reload

# os.environ["NUMBA_DISABLE_JIT"] = "1"

# number of columns of U and V must be set as env var before import dist_matrix
cols = 100
os.environ["COLUMNS"] = str(cols)

# other environment variables (set before importing dist_matrix)
os.environ["USE_64"] = "0"
os.environ["INLINE"] = "never"
os.environ["FASTMATH"] = "1"
os.environ["TARGET"] = "cuda"

from numba.cuda.kernels.device.helper import Timer  # noqa
from numba.cuda.testing import unittest, CUDATestCase  # noqa
import numba.cuda.kernels.dist_matrix  # noqa

# to overwrite env vars (source: https://stackoverflow.com/a/1254379/13697228)
reload(numba.cuda.kernels.dist_matrix)
dist_matrix = numba.cuda.kernels.dist_matrix.dist_matrix

verbose_test = True


class TestDistMat(CUDATestCase):
    """Test distance matrix calculations on GPU for various metrics."""

    def test_dist_matrix(self):
        """
        Loop through distance metrics and perform unit tests.

        The four test cases are:
            pairwise distances within a single set of vectors
            pairwise distances between two sets of vectors
            sparse pairwise distances within a single set of vectors
            sparse pairwise distances between two sets of vectors

        The ground truth for Euclidean comes from cdist.
        The ground truth for Earth Mover's distance (1-Wasserstein) is via
        a scipy.stats function.

        Helper functions are used to generate test data and support the use of
        Wasserstein distances in cdist.

        Returns
        -------
        None.

        """

        def test_data(rows, cols):
            """
            Generate seeded test values and weights for two distributions.

            Returns
            -------
            U : 2D array
                Values of first distribution.
            V : 2D array
                Values of second distribution.
            U_weights : 2D array
                Weights of first distribution.
            V_weights : 2D array
                Weights of second distribution.

            """
            np.random.seed(42)
            [U, V, U_weights, V_weights] = [
                np.random.rand(rows, cols) for i in range(4)]
            return U, V, U_weights, V_weights

        def my_wasserstein_distance(u_uw, v_vw):
            """
            Return Earth Mover's distance using concatenated values and weights.

            Parameters
            ----------
            u_uw : 1D numeric array
                Horizontally stacked values and weights of first distribution.
            v_vw : TYPE
                Horizontally stacked values and weights of second distribution.

            Returns
            -------
            distance : numeric scalar
                Earth Mover's distance given two distributions.

            """
            # split into values and weights
            n = len(u_uw)
            i = n // 2
            u = u_uw[0: i]
            uw = u_uw[i: n]
            v = v_vw[0: i]
            vw = v_vw[i: n]
            # calculate distance
            distance = scipy_wasserstein_distance(
                u, v, u_weights=uw, v_weights=vw)
            return distance

        def join_wasserstein(U, V, Uw, Vw):
            """
            Horizontally stack values and weights for each distribution.

            Weights are added as additional columns to values.

            Example:
                u_uw, v_vw = join_wasserstein(u, v, uw, vw)
                d = my_wasserstein_distance(u_uw, v_vw)
                cdist(u_uw, v_vw, metric=my_wasserstein_distance)

            Parameters
            ----------
            u : 1D or 2D numeric array
                First set of distribution values.
            v : 1D or 2D numeric array
                Second set of values of distribution values.
            uw : 1D or 2D numeric array
                Weights for first distribution.
            vw : 1D or 2D numeric array
                Weights for second distribution.

            Returns
            -------
            u_uw : 1D or 2D numeric array
                Horizontally stacked values and weights of first distribution.
            v_vw : TYPE
                Horizontally stacked values and weights of second distribution.

            """
            U_Uw = np.concatenate((U, Uw), axis=1)
            V_Vw = np.concatenate((V, Vw), axis=1)
            return U_Uw, V_Vw

        tol = 1e-5

        # test and production data
        pairs = np.array([(0, 1), (1, 2), (2, 3)])
        i, j = pairs[0]

        for testQ in [True, False]:
            # large # of rows with testQ == True is slow due to use of
            # cdist and non-jitted scipy-wasserstein
            if testQ:
                rows = 6
            else:
                rows = 1000
            print("====[PARTIAL TEST WITH {} ROWS]====".format(rows))
            U, V, U_weights, V_weights = test_data(rows, cols)
            Utest, Vtest, Uwtest, Vwtest = [x[0:6] for x in [U, V, U_weights, V_weights]]  # noqa

            for target in ["cuda"]:  # "cpu" not implemented
                for metric in ['euclidean', 'wasserstein']:
                    print('[' + target.upper() + "_" + metric.upper() + ']')

                    if testQ:
                        # compile
                        dist_matrix(Utest, U_weights=Uwtest, metric=metric)
                        with Timer("one set"):
                            one_set = dist_matrix(
                                U, U_weights=U_weights, metric=metric)  # noqa
                            if verbose_test:
                                print(one_set, '\n')

                    # compile
                    dist_matrix(Utest, V=Vtest, U_weights=Uwtest,
                                V_weights=Vwtest, metric=metric)
                    with Timer("two set"):
                        two_set = dist_matrix(U, V=V, U_weights=U_weights,
                                              V_weights=V_weights, metric=metric)  # noqa
                        if testQ and verbose_test:
                            print(two_set, '\n')

                        one_set_sparse = dist_matrix(
                            U, U_weights=U_weights, pairs=pairs, metric=metric)  # noqa
                        if testQ and verbose_test:
                            print(one_set_sparse, '\n')

                    two_set_sparse = dist_matrix(
                        U,
                        V=V,
                        U_weights=U_weights,
                        V_weights=V_weights,
                        pairs=pairs,
                        metric=metric,
                    )
                    if testQ and verbose_test:
                        print(two_set_sparse, '\n')

                    if testQ:
                        if metric == "euclidean":
                            with Timer("one set check (cdist)"):
                                one_set_check = cdist(U, U)
                            with Timer("two set check (cdist)"):
                                two_set_check = cdist(U, V)

                            one_sparse_check = [
                                euclidean(U[i], U[j]) for i, j in pairs]
                            two_sparse_check = [
                                euclidean(U[i], V[j]) for i, j in pairs]

                        elif metric == "wasserstein":
                            U_Uw, V_Vw = join_wasserstein(
                                U, V, U_weights, V_weights)

                            with Timer("one set check (cdist)"):
                                one_set_check = cdist(
                                    U_Uw, U_Uw, metric=my_wasserstein_distance)
                            with Timer("two set check (cdist)"):
                                two_set_check = cdist(
                                    U_Uw, V_Vw, metric=my_wasserstein_distance)

                            one_sparse_check = [my_wasserstein_distance(U_Uw[i], U_Uw[j])  # noqa
                                                for i, j in pairs]
                            two_sparse_check = [my_wasserstein_distance(U_Uw[i], V_Vw[j])  # noqa
                                                for i, j in pairs]

                        # check results
                        assert_allclose(
                            one_set.ravel(), one_set_check.ravel(), rtol=tol,
                            err_msg="one set {} {} distance matrix inaccurate".format(target, metric))  # noqa
                        assert_allclose(
                            two_set.ravel(), two_set_check.ravel(), rtol=tol,
                            err_msg="two set {} {} distance matrix inaccurate".format(target, metric))  # noqa
                        assert_allclose(
                            one_set_sparse, one_sparse_check, rtol=tol,
                            err_msg="one set {} {} sparse distance matrix inaccurate".format(target, metric))  # noqa
                        assert_allclose(
                            two_set_sparse, two_sparse_check, rtol=tol,
                            err_msg="two set {} {} distance matrix inaccurate".format(target, metric))  # noqa
                    elif metric == "euclidean":
                        with Timer("two set check (cdist)"):
                            two_set_check = cdist(U, V)


if __name__ == '__main__':
    unittest.main()


# TF = [
#     np.allclose(one_set, one_set_check),
#     np.allclose(two_set, two_set_check),
#     np.allclose(one_set_sparse, one_sparse_check),
#     np.allclose(two_set_sparse, two_sparse_check)
#     ]
