#! /usr/bin/env python
# ______________________________________________________________________
'''test_cfg

Test some of the features of the ControlFlowGraph class.
'''
# ______________________________________________________________________

import numba.cfg as cfg

import unittest

# ______________________________________________________________________

class TestCFG(unittest.TestCase):
    def test_loop_1(self):
        test_cfg = cfg.ControlFlowGraph()
        for block_num in xrange(6):
            test_cfg.add_block(block_num, None)
        test_cfg.add_edge(0,1)
        test_cfg.add_edge(1,2)
        test_cfg.add_edge(2,3)
        test_cfg.add_edge(2,4)
        test_cfg.add_edge(4,1)
        test_cfg.add_edge(1,5)
        test_cfg.blocks_reads[1] = set((2,3))
        test_cfg.blocks_reads[2] = set((0,1,4,5,6))
        test_cfg.blocks_reads[3] = set((3,))
        test_cfg.blocks_reads[4] = set((3,))
        test_cfg.blocks_writes[0] = set((0,1,2,3,4,5))
        test_cfg.blocks_writes[2] = set((4,5,6))
        test_cfg.blocks_writes[4] = set((3,))
        doms, reaching = test_cfg.compute_dataflow()
        self.assertEqual(doms, {0 : set((0,)),
                                1 : set((0,1)),
                                2 : set((0,1,2)),
                                3 : set((0,1,2,3)),
                                4 : set((0,1,2,4)),
                                5 : set((0,1,5))})
        self.assertEqual(tuple(test_cfg.idom(block) for block in xrange(6)),
                         (None, 0, 1, 2, 2, 1))
        self.assertEqual(test_cfg.nreaches(1), {0 : 1,
                                                1 : 1,
                                                2 : 1,
                                                3 : 2,
                                                4 : 2,
                                                5 : 2,
                                                6 : 1})
        self.assertEqual(test_cfg.phi_needed(1), set((3,4,5)))
        self.assertEqual(test_cfg.get_reaching_definitions(1),
                         {0 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0},
                          4 : {0 : 0, 1 : 0, 2 : 0, 3 : 4, 4 : 2, 5 : 2,
                               6 : 2}})

    def test_loop_2(self):
        test_cfg = cfg.ControlFlowGraph()
        for block_num in xrange(7):
            test_cfg.add_block(block_num, None)
        test_cfg.add_edge(0,6)
        test_cfg.add_edge(6,1)
        test_cfg.add_edge(1,2)
        test_cfg.add_edge(2,3)
        test_cfg.add_edge(2,4)
        test_cfg.add_edge(4,1)
        test_cfg.add_edge(1,5)
        test_cfg.blocks_reads[1] = set((2,3))
        test_cfg.blocks_reads[2] = set((0,1,4,5,6))
        test_cfg.blocks_reads[3] = set((3,))
        test_cfg.blocks_reads[4] = set((3,))
        test_cfg.blocks_writes[0] = set((0,1,2,3,4,5))
        test_cfg.blocks_writes[2] = set((4,5,6))
        test_cfg.blocks_writes[4] = set((3,))
        doms, reaching = test_cfg.compute_dataflow()
        self.assertEqual(doms, {0 : set((0,)),
                                1 : set((0,1,6)),
                                2 : set((0,1,2,6)),
                                3 : set((0,1,2,3,6)),
                                4 : set((0,1,2,4,6)),
                                5 : set((0,1,5,6)),
                                6 : set((0,6))})
        self.assertEqual(tuple(test_cfg.idom(block) for block in xrange(7)),
                         (None, 6, 1, 2, 2, 1, 0))
        self.assertEqual(test_cfg.nreaches(1), {0 : 1,
                                                1 : 1,
                                                2 : 1,
                                                3 : 2,
                                                4 : 2,
                                                5 : 2,
                                                6 : 1})
        self.assertEqual(test_cfg.phi_needed(1), set((3,4,5)))
        self.assertEqual(test_cfg.get_reaching_definitions(1),
                         {6 : {0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0},
                          4 : {0 : 0, 1 : 0, 2 : 0, 3 : 4, 4 : 2, 5 : 2,
                               6 : 2}})

    def test_loop_3(self):
        '''Test control flow analysis of a nested loop.  Hand modeled after
        test_while.while_loop_fn_5.'''
        test_cfg = cfg.ControlFlowGraph()
        for block_num in xrange(7):
            test_cfg.add_block(block_num)
        test_cfg.add_edge(0, 1)
        test_cfg.add_edge(1, 2)
        test_cfg.add_edge(1, 6)
        test_cfg.add_edge(2, 3)
        test_cfg.add_edge(3, 4)
        test_cfg.add_edge(3, 5)
        test_cfg.add_edge(4, 3)
        test_cfg.add_edge(5, 1)
        test_cfg.blocks_reads[1] = set((1,2))
        test_cfg.blocks_reads[3] = set((0,4))
        test_cfg.blocks_reads[4] = set((2,3,4))
        test_cfg.blocks_reads[5] = set((2,))
        test_cfg.blocks_reads[6] = set((3,))
        test_cfg.blocks_writes[0] = set((0,1,2,3))
        test_cfg.blocks_writes[2] = set((4,))
        test_cfg.blocks_writes[4] = set((3,4))
        test_cfg.blocks_writes[5] = set((2,))
        doms, reaching = test_cfg.compute_dataflow()
        comparison_dataflow_result = {0 : set((0,)),
                                      1 : set((0,1)),
                                      2 : set((0,1,2)),
                                      3 : set((0,1,2,3)),
                                      4 : set((0,1,2,3,4)),
                                      5 : set((0,1,2,3,5)),
                                      6 : set((0,1,6))}
        self.assertEqual(doms, comparison_dataflow_result)
        whole_loop = set((1,2,3,4,5))
        for loop_member in whole_loop:
            comparison_dataflow_result[loop_member].update(whole_loop)
        comparison_dataflow_result[6].update(whole_loop)
        self.assertEqual(reaching, comparison_dataflow_result)

    def test_branch_1(self):
        test_cfg = cfg.ControlFlowGraph()
        for block_num in xrange(4):
            test_cfg.add_block(block_num, None)
        test_cfg.add_edge(0,1)
        test_cfg.add_edge(0,2)
        test_cfg.add_edge(1,3)
        test_cfg.add_edge(2,3)
        test_cfg.blocks_reads[0] = set((0,))
        test_cfg.blocks_reads[3] = set((1,))
        test_cfg.blocks_writes[0] = set((0,))
        test_cfg.blocks_writes[1] = set((1,))
        test_cfg.blocks_writes[2] = set((1,))
        doms, reaching = test_cfg.compute_dataflow()
        self.assertEqual(doms, {0 : set((0,)),
                                1 : set((0,1)),
                                2 : set((0,2)),
                                3 : set((0,3))})
        idoms = tuple(test_cfg.idom(block) for block in xrange(4))
        self.assertEqual(idoms[:-1], (None, 0, 0))
        self.assertIn(idoms[-1], (1, 2))
        self.assertEqual(test_cfg.nreaches(3), {0 : 1, 1 : 2})
        self.assertEqual(test_cfg.phi_needed(3), set((1,)))
        self.assertEqual(test_cfg.get_reaching_definitions(3),
                         {1 : {0 : 0, 1 : 1},
                          2 : {0 : 0, 1 : 2}})

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_cfg.py
