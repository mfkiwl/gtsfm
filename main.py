import os
from loader.lund_dataset_loader import LundDatasetLoader
from frontend.frontend_wrapper import FrontEndWrapper
from frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from frontend.matcher_verifier.combination_matcher_verifier import CombinationMatcherVerifier
from frontend.matcher.twoway_matcher import TwoWayMatcher
from frontend.verifier.dummy_verifier import DummyVerifier


class GTSFMMain():
    def __init__(self, scene_graph, frontend, post, ra, ta):
        pass

    def create_computation_graph(self):
        g1 = scene_graph.create_computation_graph()


def main():
    loader = LundDatasetLoader(os.path.join('data', 'lund', 'door'))

    frontend = FrontEndWrapper(
        SIFTDetectorDescriptor(),
        CombinationMatcherVerifier(
            TwoWayMatcher,
            DummyVerifier()
        )
    )

    frontend_post = FrontEndPostProcessor()

    rotation_averaging = ShonanRotationAveraging()
    translation_averaging = TranslationAveraging1DSFM()

    frontend_graph = frontend.create_computation_graph(loader)

    relative_rotations_graph, relative_translations_graph = \
        frontend_post.create_computation_graph(
            frontend_graph, loader.intrinsics_graph())

    ra_results = rotation_averaging.create_computation_graph(
        len(loader), relative_rotations_graph)

    ta_results = translation_averaging.create_computation_graph(
        len(loader), relative_translations_graph, ra_results
    )

    with client:
        # temp = dask.compute(shonan_input)[0]
        final_results = dask.compute(ta_results)[0]

    print(final_results)


def alternate():
    scene_graph = SceneGraph()

    frontend = FrontEndWrapper(
        SIFTDetectorDescriptor(),
        CombinationMatcherVerifier(
            TwoWayMatcher,
            DummyVerifier()
        )
    )

    frontend_post = FrontEndPostProcessor()

    rotation_averaging = ShonanRotationAveraging()
    translation_averaging = TranslationAveraging1DSFM()

    scene_graph.create_computation_graph(
        frontend,
        frontend_post,
        rotation_averaging,
        translation_averaging
    )
