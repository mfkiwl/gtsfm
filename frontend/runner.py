"""Runs the front-end on a dataset.

Authors: Ayush Baid
"""
import os

import dask
from dask.distributed import Client

from frontend.detector_descriptor.rootsift import RootSIFTDetectorDescriptor
from frontend.frontend_wrapper import FrontEndWrapper
from frontend.matcher.twoway_withratiotest_matcher import \
    TwoWayWithRatioTestMatcher
from frontend.matcher_verifier.combination_matcher_verifier import \
    CombinationMatcherVerifier
from frontend.verifier.ransac import RANSAC
from loader.lund_dataset_loader import LundDatasetLoader
from utils.visualizations import visualize_matches

if __name__ == '__main__':

    loader = LundDatasetLoader(os.path.join('data', 'lund', 'door'))

    frontend = FrontEndWrapper(
        RootSIFTDetectorDescriptor(),
        CombinationMatcherVerifier(
            TwoWayWithRatioTestMatcher(ratio_test_threshold=0.8),
            RANSAC()
        )
    )

    # results = frontend.run_loader(loader)

    frontend_graph = frontend.create_computation_graph(loader)

    dask.visualize(*(frontend_graph.values()))

    client = Client(threads_per_worker=1, n_workers=1, memory_limit='4GB')
    print('Starting')
    with client:
        results = dask.compute(frontend_graph)[0]

    # client.shutdown()

    for ids, result_tuple in results.items():
        im1 = loader.get_image(ids[0])
        im2 = loader.get_image(ids[1])

        visualize_matches(im1.image_array,
                          im2.image_array,
                          result_tuple[1],
                          result_tuple[2],
                          file_name='sample_{}_{}.jpg'.format(ids[0], ids[1]),
                          match_width=True
                          )
