import argparse
import collections
import os
import os.path
import re
import sys
import logging

import cv2
import metrics
import numpy as np
import torch
from common import get_basic_input_transform

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GenericLearnedDescriptorExtractor:
    def __init__(self, patch_size, model, batch_size, transform=None, device="cuda"):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.model = model

        if transform is not None:
            self.transform = transform

        self.model = self.model.to(self.device)

    def __call__(self, patches):
        n_batches = (patches.size(0) + self.batch_size - 1) // self.batch_size
        descs = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx in range(n_batches):
                s = batch_idx * self.batch_size
                e = min((batch_idx + 1) * self.batch_size, patches.size(0))
                batch_data = patches[s:e, ...].to(self.device)

                desc = self.model(batch_data).to("cpu").detach().numpy()
                descs.append(desc)

        descs = np.concatenate(descs, axis=0)

        return descs


class DescriptorEvaluator(object):
    def __init__(self, extractor, datasets, batch_size, binarize=False):
        self.loader = DataLoader(
            datasets,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
        )
        self.extractor = extractor
        self.binarize = binarize
        if self.binarize:
            logger.info("descriptor evaluator is set to binary mode")

    def run(self):
        labels = []
        descs_a = []
        descs_b = []

        for patches_a, patches_b, label in tqdm(self.loader):
            descs_a.append(self.extractor(patches_a))
            descs_b.append(self.extractor(patches_b))
            labels += label.tolist()

        self.labels = np.stack(labels)
        self.descs_a = np.concatenate(descs_a, axis=0)
        self.descs_b = np.concatenate(descs_b, axis=0)

        if self.binarize:
            self.descs_a = (self.descs_a > 0).astype(np.float32)
            self.descs_b = (self.descs_b > 0).astype(np.float32)

        self.dist = (((self.descs_a - self.descs_b) ** 2).sum(axis=1) + 1e-8) ** 0.5
        self.scores = 1.0 / (self.dist + 1e-8)

    def computeROC(self):
        fpr, tpr, auc = metrics.roc(self.scores, self.labels)
        logger.info(f"Area under ROC: {auc}")
        return fpr, tpr, auc

    def computeFPR95(self):
        fpr95 = metrics.fpr95(self.labels, self.scores)
        logger.info(f"FPR95: {fpr95 * 100}%")
        return fpr95

    def computePR(self):
        precision, recall, auc = metrics.pr(self.scores, self.labels)
        logger.info(f"Area under PR: {auc}")
        return precision, recall, auc
