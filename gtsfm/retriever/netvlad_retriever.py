"""Retriever implementation which provides a NetVLAD global image descriptor to suggest potential image pairs.

Note: Similarity computation based off of Paul-Edouard Sarlin's HLOC:
Reference: https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/pairs_from_retrieval.py
https://openaccess.thecvf.com/content_cvpr_2016/papers/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.pdf

Authors: John Lambert
"""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import gtsfm.utils.logger as logger_utils
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase
from gtsfm.frontend.cacher.global_descriptor_cacher import GlobalDescriptorCacher
from gtsfm.frontend.global_descriptor.netvlad_global_descriptor import NetVLADGlobalDescriptor


logger = logger_utils.get_logger()

PLOT_SAVE_DIR = Path(__file__).parent.parent.parent / "plots"

MAX_NUM_IMAGES = 10000


class NetVLADRetriever(RetrieverBase):
    def __init__(self, num_matched: int, blocksize: int = 10) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
            blocksize: size of matching sub-blocks when creating similarity matrix.
        """
        self._num_matched = num_matched
        self._global_descriptor_model = GlobalDescriptorCacher(global_descriptor_obj=NetVLADGlobalDescriptor())
        self._blocksize = blocksize

    def run(
        self, loader: LoaderBase, visualize: bool = True
    ) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        num_images = len(loader)

        sim = self.compute_similarity_matrix(loader, num_images)

        query_names = loader.image_filenames()
        # Avoid self-matching and disallow lower triangular portion
        is_invalid_mat = ~np.triu(np.ones((num_images, num_images), dtype=bool))
        np.fill_diagonal(a=is_invalid_mat, val=True)
        pairs = pairs_from_score_matrix(sim, invalid=is_invalid_mat, num_select=self._num_matched, min_score=0)

        if visualize:
            plt.imshow(np.triu(sim.detach().cpu().numpy()))
            os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
            plt.title("Image Similarity Matrix")
            plt.savefig(os.path.join(PLOT_SAVE_DIR, "netvlad_similarity_matrix.jpg"), dpi=500)

        named_pairs = [(query_names[i], query_names[j]) for i, j in pairs]
        logger.info(f"Found %d pairs from the NetVLAD Retriever.", len(pairs))
        logger.info("Image Name Pairs:" + str(named_pairs))
        return pairs

    def compute_similarity_matrix(self, loader: LoaderBase, num_images: int) -> torch.Tensor:
        """Compute a similarity matrix between all pairs of images.

        We use block matching, to avoid excessive memory usage.
        We cannot fit more than 50x50 sized block into memory, on a 16 GB RAM machine.

        A similar blocked exhaustive matching implementation can be found in COLMAP:
        https://github.com/colmap/colmap/blob/dev/src/feature/matching.cc#L899

        Returns:
            sim: tensor of shape (num_images, num_images) representing similarity matrix.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if num_images > MAX_NUM_IMAGES:
            raise RuntimeError("Cannot construct similarity matrix of this size.")

        sim = torch.zeros((num_images, num_images))
        num_blocks = math.ceil(num_images / self._blocksize)

        for block_i in range(num_blocks):
            for block_j in range(num_blocks):
                # only compute the upper triangular portion of the similarity matrix.
                if block_i > block_j:
                    continue
                logger.info("Computing matching block (%d/%d,%d/%d)", block_i, num_blocks - 1, block_j, num_blocks - 1)
                block_i_query_descs = []
                block_j_query_descs = []

                i_start = block_i * self._blocksize
                i_end = (block_i + 1) * self._blocksize
                i_end = min(i_end, num_images)

                j_start = block_j * self._blocksize
                j_end = (block_j + 1) * self._blocksize
                j_end = min(j_end, num_images)

                block_i_idxs = np.arange(i_start, i_end)
                block_j_idxs = np.arange(j_start, j_end)

                for i in block_i_idxs:
                    image = loader.get_image(i)
                    block_i_query_descs.append(self._global_descriptor_model.describe(image))

                for j in block_j_idxs:
                    image = loader.get_image(j)
                    block_j_query_descs.append(self._global_descriptor_model.describe(image))

                # Form (K,D) for K images.
                block_i_query_descs = torch.from_numpy(np.array(block_i_query_descs))
                block_j_query_descs = torch.from_numpy(np.array(block_j_query_descs))

                # Einsum equivalent to (img_descs @ img_descs.T)
                sim_block = torch.einsum("id,jd->ij", block_i_query_descs.to(device), block_j_query_descs.to(device))
                sim[i_start:i_end, j_start:j_end] = sim_block

        return sim


def pairs_from_score_matrix(
    scores: torch.Tensor, invalid: np.array, num_select: int, min_score: Optional[float] = None
) -> List[Tuple[int, int]]:
    """Identify image pairs from a score matrix.

    Args:
        scores: (K1,K2) for matching K1 images against K2 images.
        invalid: (K1,K2) boolean array indicating invalid match pairs (e.g. self-matches).
        num_select: number of matches to select, per query.
        min_score: minimum allowed similarity score.

    Returns:
        pairs: tuples representing pairs (i1,i2) of images.
    """
    N, _ = scores.shape
    # if there are only N images to choose from, selecting more than N is not allowed
    num_select = min(num_select, N)

    assert scores.shape == invalid.shape
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        # logical OR.
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, k=num_select, dim=1)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs
