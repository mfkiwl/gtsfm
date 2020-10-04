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

        iRj_dict = {
            (0, 1): Rot3.Rodrigues(0, 30*np.pi/180, 0),
            (1, 2): Rot3.Rodrigues(0, 0, 20*np.pi/180),
        }

        expected_wRi = [
            Rot3.Rodrigues(0, 0, 0),
            Rot3.Rodrigues(0, 30*np.pi/180, 0),
            Rot3.Rodrigues(0, 30*np.pi/180, 20*np.pi/180),
        ]

        computed_wRi = self.obj.run(3, iRj_dict)

        print(expected_wRi[1])
        print(computed_wRi[1].between(computed_wRi[0]))
        self.assertTrue(expected_wRi[1].equals(computed_wRi[1].between(computed_wRi[0]), 1e-5))
        self.assertTrue(expected_wRi[2].equals(computed_wRi[2].between(computed_wRi[1]), 1e-5))