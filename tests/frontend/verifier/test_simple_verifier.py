"""Tests for frontend's simple verifier.

Authors: Ayush Baid
"""
import unittest

import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.frontend.verifier.simple_verifier import SimpleVerifier


class TestSimpleVerifier(test_verifier_base.TestVerifierBase):
    """Unit tests for the SimpleVerifier.

    All unit test functions defined in TestVerifierBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.verifier = SimpleVerifier()


if __name__ == "__main__":
    unittest.main()
