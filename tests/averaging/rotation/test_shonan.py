"""Tests for Shonan rotation averaging.

Authors: Ayush Baid
"""


import numpy as np
import tests.averaging.rotation.test_rotation_averaging_base as test_rotation_averaging_base
from averaging.rotation.shonan import ShonanRotationAveraging
from gtsam import Rot3


class TestShonanRotationAveraging(
        test_rotation_averaging_base.TestRotationAveragingBase):
    """Test class for Shonan rotation averaging."""

    def setUp(self):
        super().setUp()

        self.obj = ShonanRotationAveraging()

    def test_simple(self):
        """Test a simple case with just three relative rotations."""

        rotations_iRj = {
            (0, 1): Rot3.Rodrigues(0, 30*np.pi/180, 0),
            (1, 2): Rot3.Rodrigues(0, 0, 20*np.pi/180),
        }

        expected_wRi = [
            Rot3.Rodrigues(0, 0, 0),
            Rot3.Rodrigues(0, 30*np.pi/180, 0),
            Rot3.Rodrigues(0, 30*np.pi/180, 20*np.pi/180),
        ]

        computed_wRi = self.obj.run(3, rotations_iRj)

        for computed_rot, expected_rot in zip(computed_wRi, expected_wRi):
            self.assertTrue(expected_rot.equals(computed_rot, 1e-5))
