"""
Tests for SIFT detector descriptor

Authors: Ayush Baid
"""
import tests.frontend.detector_descriptor.test_detector_descriptor_base as test_detector_descriptor_base
from frontend.detector.detector_from_joint_detector_descriptor import \
    DetectorFromDetectorDescriptor
from frontend.detector_descriptor.sift import SIFT


class TestSIFT(test_detector_descriptor_base.TestDetectorDescriptorBase):
    """
    Main test class for detector-description combination base class in frontend.

    We re-use detector specific test cases from TestDetectorBase.
    """

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = SIFT()

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(
            self.detector_descriptor)
