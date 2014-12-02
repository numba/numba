from __future__ import print_function
from numba.hsa.hsadrv.driver import hsa, Queue
import numba.unittest_support as unittest


class TestHsaAgents(unittest.TestCase):
    def test_agents_init(self):
        self.assertGreater(len(hsa.agents), 0)

    def test_agents_create_queue(self):
        for agent in hsa.agents:
            queue = agent.create_queue_single(2 ** 5)
            self.assertIsInstance(queue, Queue)


if __name__ == '__main__':
    unittest.main()
