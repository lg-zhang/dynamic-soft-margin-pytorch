import logging
import os.path
import random
import sys
import time

import cv2
import numpy as np
import scipy.io as sio
import torch
import torchvision
from datasets.utils import np_flip, np_rotate


logger = logging.getLogger(__name__)


def compute_start_end_from_labels(labels):
    unique_start = []
    unique_end = []

    unique_start.append(0)
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            if i - 1 == unique_start[-1]:
                unique_start.pop()  # lonely point
            else:
                unique_end.append(i - 1)
                unique_start.append(i)

    if unique_start[-1] != len(labels) - 1:
        unique_end.append(len(labels) - 1)
    else:
        unique_start.pop()

    return unique_start, unique_end


class BrownDataset(torchvision.datasets.PhotoTour):
    mean = {
        "notredame": 0.4854,
        "yosemite": 0.4844,
        "liberty": 0.4437,
        "notredame_harris": 0.4854,
        "yosemite_harris": 0.4844,
        "liberty_harris": 0.4437,
        "ND": 0.4854,
        "YO": 0.4844,
        "LI": 0.4437,
        "ND_harris": 0.4854,
        "YO_harris": 0.4844,
        "LI_harris": 0.4437,
    }

    std = {
        "notredame": 0.1864,
        "yosemite": 0.1818,
        "liberty": 0.2019,
        "notredame_harris": 0.1864,
        "yosemite_harris": 0.1818,
        "liberty_harris": 0.2019,
        "ND": 0.1864,
        "YO": 0.1818,
        "LI": 0.2019,
        "ND_harris": 0.1864,
        "YO_harris": 0.1818,
        "LI_harris": 0.2019,
    }

    length = {
        "notredame": 468159,
        "yosemite": 633587,
        "liberty": 450092,
        "liberty_harris": 379587,
        "yosemite_harris": 450912,
        "notredame_harris": 325295,
        "ND": 468159,
        "YO": 633587,
        "LI": 450092,
        "LI_harris": 379587,
        "YO_harris": 450912,
        "ND_harris": 325295,
        "HP": 2511472,
    }

    def __init__(
        self, name, data_aug=False, triplet=False, max_samples=-1, *args, **kwargs
    ):
        name = "yosemite" if name == "YO" else name
        name = "notredame" if name == "ND" else name
        name = "liberty" if name == "LI" else name
        self._data_aug = data_aug
        self._triplet = triplet
        self._max_samples = max_samples

        super(BrownDataset, self).__init__(name=name, *args, **kwargs)

        # convert to numpy
        self.labels = self.labels.numpy()
        self.data = self.data.numpy()
        self.matches = self.matches.numpy()

        # process labels
        self.unique_start, self.unique_end = compute_start_end_from_labels(self.labels)
        logger.info(f"number of unique labels = {len(self.unique_start)}")

        if self._max_samples != -1:
            logger.info(f"limiting the number of samples to {self._max_samples}")
            self.unique_start = self.unique_start[
                : min(len(self.unique_start), self._max_samples)
            ]
            self.unique_end = self.unique_end[
                : min(len(self.unique_end), self._max_samples)
            ]
            self.matches = self.matches[: min(len(self.matches), self._max_samples)]

    def __getitem__(self, index):
        """
        :param index: index in the unique 3D point list
        :return: paired-up patches
        """
        if self.train:
            if self._triplet:
                return self._get_item_triplet(index)
            return self._get_item_siamese(index)

        return self._get_test_pair(index)

    def __len__(self):
        if self.train:
            return len(self.unique_start)
        return len(self.matches)

    def _get_test_pair(self, index):
        m = self.matches[index]
        patch_a, patch_b = self.data[m[0]], self.data[m[1]]

        if self.transform is not None:
            patch_a = self.transform(patch_a)
            patch_b = self.transform(patch_b)

        return patch_a, patch_b, m[2]

    def _get_item_siamese(self, index):
        idx1, idx2 = self._get_pair_from_unique_index(index, is_match=True)

        patch_a = self.data[idx1]
        patch_p = self.data[idx2]

        if self._data_aug:
            patch_a, patch_p = np_rotate([patch_a, patch_p], random.randint(0, 3))
            patch_a, patch_p = np_flip([patch_a, patch_p], random.randint(0, 2))

        if self.transform is not None:
            patch_a = self.transform(patch_a)
            patch_p = self.transform(patch_p)

        return patch_a, patch_p

    def _get_item_triplet(self, index):
        idx1, idx2 = self._get_pair_from_unique_index(index, is_match=True)
        _, idx3 = self._get_pair_from_unique_index(index, is_match=False)

        patch_a = self.data[idx1]
        patch_p = self.data[idx2]
        patch_n = self.data[idx3]

        if self._data_aug:
            patch_a, patch_p, patch_n = np_rotate(
                [patch_a, patch_p, patch_n], random.randint(0, 3)
            )
            patch_a, patch_p, patch_n = np_flip(
                [patch_a, patch_p, patch_n], random.randint(0, 2)
            )

        if self.transform is not None:
            patch_a = self.transform(patch_a)
            patch_p = self.transform(patch_p)
            patch_n = self.transform(patch_n)

        return patch_a, patch_p, patch_n

    def _get_pair_from_unique_index(self, index, is_match):
        """
        :param index: index in the unique 3D point list
        :param is_match: whether this pair is a match
        :return: a pair of indices
        """
        s = self.unique_start[index]
        e = self.unique_end[index]

        if is_match:
            idx1, idx2 = np.random.choice(np.arange(s, e + 1), 2, replace=False)
            return idx1, idx2

        idx1 = random.randint(s, e)
        index2 = index
        while index2 == index:
            index2 = random.randint(0, len(self.unique_start) - 1)
        idx2 = random.randint(self.unique_start[index2], self.unique_end[index2])
        return idx1, idx2

    def _get_pair_using_patch_index(self, p_index):
        s = self.unique_start[self.labels[p_index]]
        e = self.unique_end[self.labels[p_index]]

        avail = range(s, p_index) + range(p_index + 1, e + 1)
        p_index2 = avail[random.randint(0, len(avail) - 1)]
        return p_index, p_index2

    def export_mfile(self, out_p=None):
        """ Export test data for evaluation
        :param out_p: output .mat file path
        :return: None
        """
        data1_idx = [m[0] for m in self.matches]
        data2_idx = [m[1] for m in self.matches]
        labels = np.array([m[2] for m in self.matches])
        logger.info("obtained match indices")

        logger.info("converting to numpy array...")
        data1, data2 = self.data[data1_idx].numpy(), self.data[data2_idx].numpy()
        logger.info("conversion done")

        logger.info("concatenating...")
        patches = np.concatenate((data1, data2), axis=0)
        logger.info("concatenation done")

        if out_p is not None:
            sio.savemat(out_p, {"patches": patches, "labels": labels})

        return patches, labels

    def from_mfile(self, mfile_path):
        raise NotImplementedError
