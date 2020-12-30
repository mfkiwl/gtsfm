"""Tests for frontend's Ransac verifier.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.ransac import Ransac


class TestRansac(test_verifier_base.TestVerifierBase):
    """Unit tests for the Ransac verifier.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.verifier = Ransac()

    def test_simple_scene_large(self):
        """Test a simple scene with 200 points, 100 on each plane."""

        # overriding because simple ransac w/ 5 point will pass for degenerate
        # case but w/ 8 point will fail.

        self.assertTrue(
            self.verify_scene_with_two_planes(
                100, 100, use_exact_intrinsics=True
            )
        )

        self.assertFalse(
            self.verify_scene_with_two_planes(
                100, 100, use_exact_intrinsics=False
            )
        )

    def test_lopsided_scene_extralarge(self):
        """Test a lopsided with 2000 points, 1995 on 1st plane and 5 on 2nd."""

        # overriding because F-matrix w/ 8-point algorithm will fail.
        self.assertTrue(
            self.verify_scene_with_two_planes(
                1995, 5, use_exact_intrinsics=True
            )
        )

        self.assertFalse(
            self.verify_scene_with_two_planes(
                1995, 5, use_exact_intrinsics=False
            )
        )

    def test_planar_scene_large(self):
        """Test a lopsided with 100 points, all on a plane."""

        # overriding because simple ransac w/ 5 point will pass.

        self.assertTrue(
            self.verify_scene_with_two_planes(100, 0, use_exact_intrinsics=True)
        )

        self.assertFalse(
            self.verify_scene_with_two_planes(
                100, 0, use_exact_intrinsics=False
            )
        )


if __name__ == "__main__":
    unittest.main()
