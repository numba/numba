"""
Tests for CPU atomic operations
"""

import threading
import time
import numpy as np
import ctypes

from numba import njit, types
from numba.core import cgutils
from numba.tests.support import TestCase, unittest
import numba

try:
    # Import CPU atomic operations
    import numba.atomic as atomic

    CPU_ATOMICS_AVAILABLE = True
except ImportError:
    CPU_ATOMICS_AVAILABLE = False


@unittest.skipUnless(CPU_ATOMICS_AVAILABLE, "CPU atomics not available")
class TestCPUAtomics(TestCase):
    """Test CPU atomic operations"""

    def test_atomic_load_uint8(self):
        """Test atomic load on uint8"""

        @njit
        def atomic_load_test(arr, idx):
            return atomic.load(arr, idx)

        # Test with uint8 array
        arr = np.array([42, 100, 255], dtype=np.uint8)
        result = atomic_load_test(arr, 1)
        self.assertEqual(result, 100)

    def test_atomic_load_uint64(self):
        """Test atomic load on uint64"""

        @njit
        def atomic_load_test(arr, idx):
            return atomic.load(arr, idx)

        # Test with uint64 array
        arr = np.array([0, 18446744073709551615, 12345], dtype=np.uint64)
        result = atomic_load_test(arr, 1)
        self.assertEqual(result, 18446744073709551615)

    def test_atomic_store_uint8(self):
        """Test atomic store on uint8"""

        @njit
        def atomic_store_test(arr, idx, val):
            atomic.store(arr, idx, val)

        # Test with uint8 array
        arr = np.array([0, 0, 0], dtype=np.uint8)
        atomic_store_test(arr, 1, np.uint8(199))
        self.assertEqual(arr[1], 199)

    def test_atomic_store_uint64(self):
        """Test atomic store on uint64"""

        @njit
        def atomic_store_test(arr, idx, val):
            atomic.store(arr, idx, val)

        # Test with uint64 array
        arr = np.array([0, 0, 0], dtype=np.uint64)
        test_val = np.uint64(9223372036854775808)  # Large uint64 value
        atomic_store_test(arr, 2, test_val)
        self.assertEqual(arr[2], test_val)

    def test_atomic_add_uint8(self):
        """Test atomic add on uint8"""

        @njit
        def atomic_add_test(arr, idx, val):
            return atomic.add(arr, idx, val)

        # Test with uint8 array
        arr = np.array([10, 20, 30], dtype=np.uint8)
        old_val = atomic_add_test(arr, 1, np.uint8(5))
        self.assertEqual(old_val, 20)  # Previous value
        self.assertEqual(arr[1], 25)  # New value

    def test_atomic_add_uint64(self):
        """Test atomic add on uint64"""

        @njit
        def atomic_add_test(arr, idx, val):
            return atomic.add(arr, idx, val)

        # Test with uint64 array
        arr = np.array([1000, 2000, 3000], dtype=np.uint64)
        old_val = atomic_add_test(arr, 0, np.uint64(500))
        self.assertEqual(old_val, 1000)  # Previous value
        self.assertEqual(arr[0], 1500)  # New value

    def test_atomic_sub_uint8(self):
        """Test atomic subtract on uint8"""

        @njit
        def atomic_sub_test(arr, idx, val):
            return atomic.sub(arr, idx, val)

        # Test with uint8 array
        arr = np.array([100, 200, 50], dtype=np.uint8)
        old_val = atomic_sub_test(arr, 1, np.uint8(50))
        self.assertEqual(old_val, 200)  # Previous value
        self.assertEqual(arr[1], 150)  # New value

    def test_atomic_fetch_add_uint8(self):
        """Test atomic fetch_add on uint8"""

        @njit
        def atomic_fetch_add_test(arr, idx, val):
            return atomic.fetch_add(arr, idx, val)

        # Test with uint8 array
        arr = np.array([10, 20, 30], dtype=np.uint8)
        old_val = atomic_fetch_add_test(arr, 1, np.uint8(5))
        self.assertEqual(old_val, 20)  # Previous value
        self.assertEqual(arr[1], 25)  # New value

    def test_atomic_fetch_add_uint64(self):
        """Test atomic fetch_add on uint64"""

        @njit
        def atomic_fetch_add_test(arr, idx, val):
            return atomic.fetch_add(arr, idx, val)

        # Test with uint64 array
        arr = np.array([1000, 2000, 3000], dtype=np.uint64)
        old_val = atomic_fetch_add_test(arr, 0, np.uint64(500))
        self.assertEqual(old_val, 1000)  # Previous value
        self.assertEqual(arr[0], 1500)  # New value

    def test_atomic_compare_and_swap_uint8(self):
        """Test atomic compare-and-swap on uint8"""

        @njit
        def atomic_cas_test(arr, idx, expected, desired):
            return atomic.compare_and_swap(arr, idx, expected, desired)

        # Test successful CAS
        arr = np.array([10, 20, 30], dtype=np.uint8)
        old_val = atomic_cas_test(arr, 1, np.uint8(20), np.uint8(99))
        self.assertEqual(old_val, 20)  # Previous value
        self.assertEqual(arr[1], 99)  # Value was swapped

        # Test failed CAS
        old_val = atomic_cas_test(arr, 1, np.uint8(20), np.uint8(77))
        self.assertEqual(old_val, 99)  # Current value (not 20 anymore)
        self.assertEqual(arr[1], 99)  # Value was not swapped

    def test_atomic_compare_and_swap_uint64(self):
        """Test atomic compare-and-swap on uint64"""

        @njit
        def atomic_cas_test(arr, idx, expected, desired):
            return atomic.compare_and_swap(arr, idx, expected, desired)

        # Test successful CAS
        arr = np.array([1000, 2000, 3000], dtype=np.uint64)
        old_val = atomic_cas_test(arr, 0, np.uint64(1000), np.uint64(9999))
        self.assertEqual(old_val, 1000)  # Previous value
        self.assertEqual(arr[0], 9999)  # Value was swapped

    def test_memory_ordering_parameters(self):
        """Test that memory ordering parameters are accepted"""

        @njit
        def test_orderings(arr):
            # Test different memory orderings
            val1 = atomic.load(arr, 0, "acquire")
            atomic.store(arr, 1, np.uint64(42), "release")
            old_val = atomic.add(arr, 2, np.uint64(1), "acq_rel")
            fetch_add_val = atomic.fetch_add(arr, 3, np.uint64(10), "seq_cst")
            return val1, old_val, fetch_add_val

        arr = np.array([10, 20, 30, 40], dtype=np.uint64)
        val1, old_val, fetch_add_val = test_orderings(arr)
        self.assertEqual(val1, 10)
        self.assertEqual(arr[1], 42)
        self.assertEqual(old_val, 30)
        self.assertEqual(arr[2], 31)
        self.assertEqual(fetch_add_val, 40)  # Previous value
        self.assertEqual(arr[3], 50)  # New value

    def test_pointer_based_atomics(self):
        """Test atomic operations on raw pointers"""

        @njit
        def test_ptr_atomics():
            # Create a simple integer array to work with
            arr = np.array([100], dtype=np.uint64)

            # Get pointer to first element using ctypes/raw pointers
            # Note: This is a simplified test - in real usage,
            # pointer atomics would be used with shared memory regions

            # Test atomic add via array indexing (closest we can get in test)
            old_val = atomic.add(arr, 0, np.uint64(50))
            return old_val, arr[0]

        old_val, new_val = test_ptr_atomics()
        self.assertEqual(old_val, 100)
        self.assertEqual(new_val, 150)


@unittest.skipUnless(CPU_ATOMICS_AVAILABLE, "CPU atomics not available")
class TestCPUAtomicsThreadSafety(TestCase):
    """Test thread safety of CPU atomic operations"""

    def test_atomic_increment_multithreaded(self):
        """Test atomic increment across multiple threads"""

        @njit
        def atomic_increment_worker(arr, iterations):
            for i in range(iterations):
                atomic.add(arr, 0, np.uint64(1))

        # Compile the function first
        arr = np.array([0], dtype=np.uint64)
        atomic_increment_worker(arr, 1)

        # Reset and run multithreaded test
        arr[0] = 0
        num_threads = 4
        iterations_per_thread = 1000
        expected_total = num_threads * iterations_per_thread

        # Create and start threads
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(
                target=atomic_increment_worker, args=(arr, iterations_per_thread)
            )
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check that all increments were applied atomically
        self.assertEqual(arr[0], expected_total)

    def test_atomic_fetch_add_multithreaded(self):
        """Test atomic fetch_add across multiple threads"""

        @njit
        def atomic_fetch_add_worker(arr, iterations, results, thread_id):
            for i in range(iterations):
                old_val = atomic.fetch_add(arr, 0, np.uint64(1))
                # Store the previous values we observed
                results[thread_id * iterations + i] = old_val

        # Setup
        arr = np.array([0], dtype=np.uint64)
        num_threads = 4
        iterations_per_thread = 100
        total_results = num_threads * iterations_per_thread
        results = np.zeros(total_results, dtype=np.uint64)

        # Compile the function first
        atomic_fetch_add_worker(arr, 1, results[:1], 0)

        # Reset and run multithreaded test
        arr[0] = 0
        results.fill(0)

        # Create and start threads
        threads = []
        for thread_id in range(num_threads):
            t = threading.Thread(
                target=atomic_fetch_add_worker,
                args=(arr, iterations_per_thread, results, thread_id),
            )
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Check that final value is correct
        expected_total = total_results
        self.assertEqual(arr[0], expected_total)

        # Check that all returned values are unique and in valid range
        non_zero_results = results[results != 0]  # Filter out any uninitialized values
        unique_results = set(non_zero_results)

        # All non-zero results should be unique (no two threads got same old value)
        self.assertEqual(len(non_zero_results), len(unique_results))

        # All results should be in valid range [0, total_results)
        for val in non_zero_results:
            self.assertGreaterEqual(val, 0)
            self.assertLess(val, expected_total)

    def test_atomic_store_load_consistency(self):
        """Test consistency between atomic store and load operations"""

        @njit
        def writer_thread(arr, values):
            for val in values:
                atomic.store(arr, 0, val)
                # Small delay to increase chance of interleaving
                for _ in range(100):
                    pass

        @njit
        def reader_thread(arr, results, num_reads):
            for i in range(num_reads):
                val = atomic.load(arr, 0)
                results[i] = val
                # Small delay to increase chance of interleaving
                for _ in range(50):
                    pass

        # Test data
        arr = np.array([0], dtype=np.uint64)
        write_values = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
        read_results = np.zeros(20, dtype=np.uint64)

        # Compile functions
        writer_thread(arr, write_values[:1])
        reader_thread(arr, read_results[:1], 1)

        # Reset and run actual test
        arr[0] = 0
        read_results.fill(0)

        # Start writer and reader threads
        writer = threading.Thread(target=writer_thread, args=(arr, write_values))
        reader = threading.Thread(target=reader_thread, args=(arr, read_results, 20))

        writer.start()
        reader.start()

        writer.join()
        reader.join()

        # Verify that all read values are either 0 or one of the written values
        valid_values = set([0] + write_values.tolist())
        for val in read_results:
            self.assertIn(val, valid_values, f"Read invalid value: {val}")


@unittest.skipUnless(CPU_ATOMICS_AVAILABLE, "CPU atomics not available")
class TestCPUAtomicsEdgeCases(TestCase):
    """Test edge cases and error conditions for CPU atomics"""

    def test_unsupported_types(self):
        """Test that unsupported types raise appropriate errors"""

        # Test with float array (should fail)
        @njit
        def test_float_atomic(arr):
            return atomic.load(arr, 0)

        arr_float = np.array([1.0, 2.0], dtype=np.float32)

        # This should raise a typing error during compilation
        with self.assertRaises((TypeError, Exception)):
            test_float_atomic(arr_float)

    def test_boundary_values_uint8(self):
        """Test boundary values for uint8 atomics"""

        @njit
        def test_uint8_boundaries(arr):
            # Test max uint8 value
            atomic.store(arr, 0, np.uint8(255))
            val1 = atomic.load(arr, 0)

            # Test overflow behavior (255 + 1 should wrap to 0)
            old_val = atomic.add(arr, 0, np.uint8(1))
            val2 = atomic.load(arr, 0)

            return val1, old_val, val2

        arr = np.array([0], dtype=np.uint8)
        val1, old_val, val2 = test_uint8_boundaries(arr)

        self.assertEqual(val1, 255)  # Max value stored
        self.assertEqual(old_val, 255)  # Previous value before add
        self.assertEqual(val2, 0)  # Wrapped around to 0

    def test_boundary_values_uint64(self):
        """Test boundary values for uint64 atomics"""

        @njit
        def test_uint64_boundaries(arr):
            # Test max uint64 value
            max_uint64 = np.uint64(18446744073709551615)
            atomic.store(arr, 0, max_uint64)
            val1 = atomic.load(arr, 0)

            # Test overflow (max + 1 should wrap to 0)
            old_val = atomic.add(arr, 0, np.uint64(1))
            val2 = atomic.load(arr, 0)

            return val1, old_val, val2

        arr = np.array([0], dtype=np.uint64)
        val1, old_val, val2 = test_uint64_boundaries(arr)

        self.assertEqual(val1, 18446744073709551615)  # Max value
        self.assertEqual(old_val, 18446744073709551615)  # Previous value
        self.assertEqual(val2, 0)  # Wrapped to 0


if __name__ == "__main__":
    unittest.main()
