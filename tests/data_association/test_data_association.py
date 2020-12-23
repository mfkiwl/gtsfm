"""Unit test for the DataAssociation class (and implicitly the Point3dInitializer class).
Triangulation examples from:
     borglab/gtsam/python/gtsam/tests/test_Triangulation.py 
     gtsam/geometry/tests/testTriangulation.cpp

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""
import unittest
from typing import Dict, List, Tuple

import dask
import numpy as np
from common.keypoints import Keypoints
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point2Vector,
    Point3,
    Pose3,
    Pose3Vector,
    Rot3,
)
from gtsam.utils.test_case import GtsamTestCase

from data_association.data_assoc import DataAssociation, TriangulationParam


def get_pose3_vector(num_poses: int) -> Pose3Vector:
    """ Generate camera poses for use in triangulation tests """

    # Looking along X-axis, 1 meter above ground plane (x-y)
    upright = Rot3.Ypr(-np.pi / 2, 0.0, -np.pi / 2)
    pose1 = Pose3(upright, Point3(0, 0, 1))

    # create second camera 1 meter to the right of first camera
    pose2 = pose1.compose(Pose3(Rot3(), Point3(1, 0, 0)))

    # Add third camera slightly rotated
    rotatedCamera = Rot3.Ypr(0.1, 0.2, 0.1)
    pose3 = pose1.compose(Pose3(rotatedCamera, Point3(0.1, -2, -0.1)))

    available_poses = [pose1, pose2, pose3]

    pose3_vec = Pose3Vector()
    for i in range(num_poses):
        pose3_vec.append(available_poses[i])
    return pose3_vec


def generate_noisy_2d_measurements(
    world_point: Point3,
    calibrations: List[Cal3Bundler],
    per_image_noise_vecs: np.ndarray,
    poses: Pose3Vector,
) -> Tuple[
    List[Keypoints],
    List[Tuple[int, int]],
    Dict[int, PinholeCameraCal3Bundler],
]:
    """
    Generate PinholeCameras from specified poses and calibrations, and then generate
    1 measurement per camera of a given 3d point.

    Args:
        world_point: 3d coords of 3d landmark in world frame
        calibrations: List of calibrations for each camera
        noise_params: List of amounts of noise to be added to each measurement
        poses: List of poses for each camera in world frame

    Returns:
        keypoints_list: List of keypoints in all images (projected measurements in all images)
        img_idxs: Tuple of indices for all images
        cameras: Dictionary mapping image index i to calibrated PinholeCamera object
    """
    keypoints_list = []
    measurements = Point2Vector()
    cameras = dict()
    for i in range(len(poses)):
        camera = PinholeCameraCal3Bundler(poses[i], calibrations[i])
        # Project landmark into two cameras and triangulate
        z = camera.project(world_point)
        cameras[i] = camera
        measurement = z + per_image_noise_vecs[i]
        measurements.append(measurement)
        keypoints_list += [Keypoints(coordinates=measurement.reshape(1, 2))]

    # Create image indices for each pose - only subsequent pairwise matches
    # assumed, e.g. between images (0,1) and images (1,2)
    img_idxs = []
    for i in range(len(poses) - 1):
        img_idxs += [(i, i + 1)]

    return keypoints_list, img_idxs, cameras


class TestDataAssociation(GtsamTestCase):
    """
    Unit tests for data association module, which maps the feature tracks to their 3D landmarks.
    """

    def setUp(self):
        """
        Set up the data association module.
        """
        super().setUp()

        # landmark ~5 meters infront of camera
        self.expected_landmark = Point3(5, 0.5, 1.2)

        # shared calibration
        f, k1, k2, u0, v0 = 1500, 0, 0, 640, 480
        self.sharedCal = Cal3Bundler(f, k1, k2, u0, v0)

    def test_ransac_sample_biased_baseline_sharedCal_2poses(self):
        """ """
        mode = TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_ransac_topk_baselines_sharedCal_2poses(self):
        """ """
        mode = TriangulationParam.RANSAC_TOPK_BASELINES
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_ransac_sample_uniform_sharedCal_2poses(self):
        """ """
        mode = TriangulationParam.RANSAC_SAMPLE_UNIFORM
        self.verify_triangulation_sharedCal_2poses(mode)

    def test_no_ransac_sharedCal_2poses(self):
        """ """
        mode = TriangulationParam.NO_RANSAC
        self.verify_triangulation_sharedCal_2poses(mode)

    def verify_triangulation_sharedCal_2poses(
        self, triangulation_mode: TriangulationParam
    ):
        """
        Tests that the triangulation is accurate for shared calibration with a
        specified triangulation mode.

        Checks whether the triangulated landmark map formed from 2 measurements is valid,
        if min track length = 3 (should be invalid)

        The noise vectors represent the amount of noise to be added to measurements.
        """
        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array([[-0.1, -0.5], [0.2, -0.3]]),
            poses=get_pose3_vector(num_poses=2),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]])}

        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=triangulation_mode,
            num_ransac_hypotheses=20,
        )
        triangulated_landmark_map = da.run(
            cameras, matches_dict, keypoints_list
        )
        # assert that we cannot obtain even 1 length-3 track if we have only 2 camera poses
        # result should be empty, since nb_measurements < min track length
        assert (
            triangulated_landmark_map.number_tracks() == 0
        ), "Failure: tracks exceed expected track length (should be 0 tracks)"

    def test_triangulation_individualCal_without_ransac(self):
        """
        Tests that the triangulation is accurate for individual camera calibration, without RANSAC-based triangulation.
        Checks if cameras and triangulated 3D point are as expected.
        """
        k1 = 0
        k2 = 0
        f, u0, v0 = 1500, 640, 480
        f_, u0_, v0_ = 1600, 650, 440
        K1 = Cal3Bundler(f, k1, k2, u0, v0)
        K2 = Cal3Bundler(f_, k1, k2, u0_, v0_)

        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[K1, K2],
            per_image_noise_vecs=np.zeros((2, 2)),
            poses=get_pose3_vector(num_poses=2),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]])}

        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=2,  # at least 2 measurements required
            mode=TriangulationParam.NO_RANSAC,
        )
        sfm_data = da.run(cameras, matches_dict, keypoints_list)
        estimated_landmark = sfm_data.track(0).point3()
        self.gtsamAssertEquals(estimated_landmark, self.expected_landmark, 1e-2)

        for i in range(sfm_data.number_cameras()):
            self.gtsamAssertEquals(sfm_data.camera(i), cameras.get(i))

    def test_ransac_sample_biased_baseline_sharedCal_3poses(self):
        """ """
        mode = TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_ransac_topk_baselines_sharedCal_3poses(self):
        """ """
        mode = TriangulationParam.RANSAC_TOPK_BASELINES
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_ransac_sample_uniform_sharedCal_3poses(self):
        """ """
        mode = TriangulationParam.RANSAC_SAMPLE_UNIFORM
        self.verify_triangulation_sharedCal_3poses(mode)

    def test_no_ransac_sharedCal_3poses(self):
        """ """
        mode = TriangulationParam.NO_RANSAC
        self.verify_triangulation_sharedCal_3poses(mode)

    def verify_triangulation_sharedCal_3poses(
        self, triangulation_mode: TriangulationParam
    ):
        """
        Tests that the triangulation is accurate for shared calibration with a
        specified triangulation mode.

        Checks whether the sfm data formed from 3 measurements is valid.
        The noise vectors represent the amount of noise to be added to measurements.
        """
        keypoints_list, _, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array(
                [[-0.1, -0.5], [-0.2, 0.3], [0.1, -0.1]]
            ),
            poses=get_pose3_vector(num_poses=3),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]]), (1, 2): np.array([[0, 0]])}

        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=triangulation_mode,
            num_ransac_hypotheses=20,
        )
        sfm_data = da.run(cameras, matches_dict, keypoints_list)

        estimated_landmark = sfm_data.track(0).point3()
        # checks if computed 3D point is as expected
        self.gtsamAssertEquals(estimated_landmark, self.expected_landmark, 1e-2)

        # checks if number of tracks are as expected, should be just 1, over all 3 cameras
        assert sfm_data.number_tracks() == 1, "more tracks than expected"
        # checks if cameras saved to result are as expected
        for i in range(sfm_data.number_cameras()):
            self.gtsamAssertEquals(sfm_data.camera(i), cameras.get(i))

    def test_create_computation_graph(self):
        """
        Tests the graph to create data association for 3 images.
        Checks if result from dask computation graph is the same as result without dask.
        """
        keypoints_list, img_idxs, cameras = generate_noisy_2d_measurements(
            world_point=self.expected_landmark,
            calibrations=[self.sharedCal, self.sharedCal, self.sharedCal],
            per_image_noise_vecs=np.array(
                [[-0.1, -0.5], [-0.2, 0.3], [0.1, -0.1]]
            ),
            poses=get_pose3_vector(num_poses=3),
        )

        # create matches
        # since there is only one measurement in each image, both assigned feature index 0
        matches_dict = {(0, 1): np.array([[0, 0]]), (1, 2): np.array([[0, 0]])}

        # Run without computation graph
        da = DataAssociation(
            reproj_error_thresh=5,  # 5 px
            min_track_len=3,  # at least 3 measurements required
            mode=TriangulationParam.RANSAC_TOPK_BASELINES,
            num_ransac_hypotheses=20,
        )
        expected_sfm_data = da.run(cameras, matches_dict, keypoints_list)

        # Run with computation graph
        delayed_sfm_data = da.create_computation_graph(
            cameras,
            matches_dict,
            keypoints_list,
        )

        with dask.config.set(scheduler="single-threaded"):
            dask_sfm_data = dask.compute(delayed_sfm_data)[0]

        assert (
            expected_sfm_data.number_tracks() == dask_sfm_data.number_tracks()
        ), "Dask not configured correctly"

        for k in range(expected_sfm_data.number_tracks()):
            assert (
                expected_sfm_data.track(k).number_measurements()
                == dask_sfm_data.track(k).number_measurements()
            ), "Dask tracks incorrect"
            # Test if the measurement in both are equal
            np.testing.assert_array_almost_equal(
                expected_sfm_data.track(k).measurement(0)[1],
                dask_sfm_data.track(k).measurement(0)[1],
                decimal=1,
                err_msg="Dask measurements incorrect",
            )
        for i in range(expected_sfm_data.number_cameras()):
            self.gtsamAssertEquals(expected_sfm_data.camera(i), cameras.get(i))


if __name__ == "__main__":
    unittest.main()