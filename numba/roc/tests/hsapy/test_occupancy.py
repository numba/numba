from __future__ import print_function, division, absolute_import

from numba import unittest_support as unittest
from numba.roc.gcn_occupancy import get_limiting_factors


class TestOccupancy(unittest.TestCase):
    def check_limits(self, inputs, expected_outputs):
        outputs = get_limiting_factors(**inputs)
        for k, expect in expected_outputs.items():
            got = getattr(outputs, k)
            if k == 'occupancy':
                self.assertAlmostEqual(got, expect, msg=k)
            else:
                self.assertEqual(got, expect, k)

    def test_limits_1(self):
        inputs = dict(group_size=400,
                      vgpr_per_workitem=139,
                      sgpr_per_wave=49)
        outputs = dict(
            allowed_wave_due_to_sgpr=10,
            allowed_wave_due_to_vgpr=1,
            allowed_wave=1,
            allowed_vgpr_per_workitem=128,
            occupancy=0,
            reasons=set(['allowed_wave_due_to_vgpr',
                         'allowed_wave',
                         'group_size']),
        )
        self.check_limits(inputs, outputs)

    def test_limits_2(self):
        inputs = dict(group_size=256,
                      vgpr_per_workitem=139,
                      sgpr_per_wave=49)
        outputs = dict(
            allowed_wave_due_to_sgpr=10,
            allowed_wave_due_to_vgpr=1,
            allowed_wave=1,
            allowed_vgpr_per_workitem=256,
            occupancy=.10,
            reasons=set(),
        )
        self.check_limits(inputs, outputs)

    def test_limits_3(self):
        inputs = dict(group_size=2048,
                      vgpr_per_workitem=16,
                      sgpr_per_wave=70)
        outputs = dict(
            allowed_wave_due_to_sgpr=7,
            allowed_wave_due_to_vgpr=16,
            allowed_wave=7,
            allowed_vgpr_per_workitem=32,
            occupancy=0,
            reasons=set(['allowed_wave_due_to_sgpr',
                         'allowed_wave',
                         'group_size']),
        )
        self.check_limits(inputs, outputs)

    def test_limits_4(self):
        inputs = dict(group_size=2048,
                      vgpr_per_workitem=32,
                      sgpr_per_wave=50)
        outputs = dict(
            allowed_wave_due_to_sgpr=10,
            allowed_wave_due_to_vgpr=8,
            allowed_wave=8,
            allowed_vgpr_per_workitem=32,
            occupancy=0,
            reasons=set(['group_size']),
        )
        self.check_limits(inputs, outputs)

    def test_limits_5(self):
        inputs = dict(group_size=4,
                      vgpr_per_workitem=128,
                      sgpr_per_wave=10)
        outputs = dict(
            allowed_wave_due_to_sgpr=51,
            allowed_wave_due_to_vgpr=2,
            allowed_wave=2,
            allowed_vgpr_per_workitem=256,
            occupancy=.1,
            reasons=set(),
        )
        self.check_limits(inputs, outputs)

    def test_limits_6(self):
        inputs = dict(group_size=4,
                      vgpr_per_workitem=257,
                      sgpr_per_wave=3)
        outputs = dict(
            allowed_wave_due_to_sgpr=170,
            allowed_wave_due_to_vgpr=0,
            allowed_wave=0,
            allowed_vgpr_per_workitem=256,
            occupancy=0,
            reasons=set(['allowed_wave_due_to_vgpr',
                         'allowed_wave']),
        )
        self.check_limits(inputs, outputs)


if __name__ == '__main__':
    unittest.main()


