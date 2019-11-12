import argparse
import datetime
import logging
import os.path
import sys
import time

import common
import cv2
import numpy as np
import torch
import torch.nn as nn
from datasets import BrownDataset, HPatches
from desc_eval import DescriptorEvaluator, GenericLearnedDescriptorExtractor
from modules import DynamicSoftMarginLoss, HardNetLoss, L2Net
import misc


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d:%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()


def add_arg(*args, **kwargs):
    kwargs["help"] = "(default: %(default)s)"
    if not kwargs.get("type", bool) == bool:
        kwargs["metavar"] = ""
    parser.add_argument(*args, **kwargs)


add_arg("--cpuonly", action="store_true")
add_arg("--bs", type=int, default=1024)
add_arg("--model_dir", type=str, default=None)
add_arg("--binary", action="store_true")
add_arg("--test_data", nargs="+", type=str, default=["brown.liberty"])
add_arg("--patch_size", type=int, default=32)

args = parser.parse_args()

# select device
device = "cpu" if args.cpuonly else "cuda"

# set up the model
model = L2Net(out_dim=256 if args.binary else 128, binary=args.binary)

assert args.model_dir is not None, "model directory not specified"
model.load_state_dict(torch.load(os.path.join(args.model_dir, "model.state_dict")))
model = model.to(device)

mean_std = torch.load(os.path.join(args.model_dir, "mean_std.pt"))
tforms = common.get_basic_input_transform(
    args.patch_size, mean_std["mean"], mean_std["std"]
)

desc_extractor = GenericLearnedDescriptorExtractor(
    patch_size=args.patch_size,
    model=model,
    batch_size=args.bs,
    transform=tforms,
    device=device,
)

evaluators = {}
for dset_dot_seq in args.test_data:
    dset_name, seq_name = dset_dot_seq.split(".")
    if dset_name == "brown":
        dset = BrownDataset(
            root="./data/brown",
            name=seq_name,
            download=True,
            train=False,
            transform=tforms,
            data_aug=False,
        )
    elif dset_name == "HP":
        raise RuntimeError("please use the offical HPatches evaluation scripts")
    else:
        raise ValueError("dataset not recognized")

    logger.info(f"adding evaluator {dset_dot_seq}")
    evaluators[dset_dot_seq] = DescriptorEvaluator(
        extractor=desc_extractor,
        datasets=dset,
        batch_size=args.bs,
        binarize=args.binary,
    )


def test():
    logger.info("running evaluation...")
    fpr95 = {}
    for dset_dot_seq, evaluator in evaluators.items():
        evaluator.run()
        fpr95[dset_dot_seq] = evaluator.computeFPR95()
    return fpr95


test_result = test()
for dset_dot_seq, fpr95 in test_result.items():
    logger.info(f"FPR95-{dset_dot_seq} = {fpr95 * 100}%")
