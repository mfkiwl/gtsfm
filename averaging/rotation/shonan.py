"""Shonan Rotation Averaging.

The algorithm was proposed in "Shonan Rotation Averaging:Global Optimality by
Surfing SO(p)^n" and is implemented by wrapping up over implementation provided
by GTSAM.

References:
- https://arxiv.org/abs/2008.02737
- https://gtsam.org/

Authors: Jing Wu, Ayush Baid
"""
from typing import Dict, List, Tuple

import gtsam
import numpy as np
from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsam import BetweenFactorPose3, Pose3, Rot3, ShonanAveraging3


class ShonanRotationAveraging(RotationAveragingBase):
    """Performs Shonan rotation averaging."""

    def __init__(self):
        self._pMin = 5
        self._pMax = 30
        # lmParams = gtsam.LevenbergMarquardtParams.CeresDefaults()
        # self._params = ShonanAveragingParameters3(lmParams)

        self.noise_model = gtsam.noiseModel.Unit.Create(3)

    def run(self,
            num_poses: int,
            relative_rotations: Dict[Tuple[int, int], Rot3]
            ) -> List[Rot3]:
        """Run the rotation averaging.

        Args:
            num_poses (int): number of poses.
            relative_rotations (Dict[Tuple[int, int], Rot3]): relative
                                                              rotations between
                                                              camera poses.

        Returns:
            List[Rot3]: global rotations for each camera pose.
        """

        between_factors = []

        for idx_pair, rotation in relative_rotations.items():
            between_factors.append(BetweenFactorPose3(
                idx_pair[0],
                idx_pair[1],
                Pose3(rotation, np.zeros(3,)),
                self.noise_model
            ))

        obj = ShonanAveraging3(between_factors)

        initial = obj.initializeRandomly()
        result_values, _ = obj.run(initial, self._pMin, self._pMax)

        return [result_values.atRot3(i) for i in range(num_poses)]
