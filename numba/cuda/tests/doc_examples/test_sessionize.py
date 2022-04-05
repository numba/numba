import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestSessionization(CUDATestCase):
    """
    Test click stream sessionization
    """

    def setUp(self):
        # Prevent output from this test showing up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_sessionize(self):
        # ex_sessionize.import.begin
        from numba import cuda
        import numpy as np
        # set the timeout to one hour
        session_timeout = np.int64(np.timedelta64('3600', 's'))
        # ex_sessionize.import.end
        
        # ex_sessionize.allocate.begin
        # generate data
        ids = cuda.to_device(
            np.array([1,1,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4])
        )
        sec = cuda.to_device(
            np.array(
                [1,2,3,5000,5001,5002,1,2,3,1,2,5000,5001,10000,10001,10002,10003,15000,150001,1,5000,50001,15000,20000,25000,25001,25002, 25003],
                dtype='datetime64[ns]').astype('int64') # cast to int64 for compatibility
        )
        # create a vector to hold the results
        results = cuda.to_device(np.zeros(len(ids)))
        # ex_sessionize.allocate.end

        # ex_sessionize.kernel.begin
        @cuda.jit
        def sessionize(
            user_id, timestamp, results
        ):
            gid = cuda.grid(1)
            size = len(user_id)

            if (gid >= size):
                return

            ### DETERMINE SESSION BOUNDARIES
            is_first_datapoint = (gid == 0)
            if not is_first_datapoint:
                new_user = (user_id[gid] != user_id[gid - 1])
                timed_out = (timestamp[gid] - timestamp[gid - 1] > session_timeout)           
                is_sess_boundary = new_user or timed_out
            else:
                is_sess_boundary = True
 
            ### DETERMINE SESSION LABELS
            # this thread marks the start of a session
            if gid < size:
                if is_sess_boundary:
                    results[gid] = gid
                    
                    # make sure all session boundaries are written
                    # before populating the session id
                    grid = cuda.cg.this_grid()                
                    grid.sync()

                    look_ahead = 1
                    # check elements 'forward' of this one 
                    # until a new session boundary is found
                    while results[gid + look_ahead] == 0:
                        results[gid + look_ahead] = gid
                        look_ahead += 1
                        # don't segfault if I'm the last thread
                        if gid + look_ahead == size - 1:
                            results[gid + look_ahead] = gid
                            break
        # ex_sessionize.kernel.end

        # ex_sessionize.launch.begin
        sessionize.forall(len(ids))(ids, sec, results)
        
        print(results.copy_to_host())
        # array([ 0.,  0.,  0.,  3.,  3.,  3.,  6.,  6.,  6.,  9.,  9., 11., 11.,
        #    13., 13., 13., 13., 17., 18., 19., 20., 21., 21., 23., 24., 24.,
        #    24., 24.])
        # ex_sessionize.launch.end

    
if __name__ == '__main__':
    unittest.main()
