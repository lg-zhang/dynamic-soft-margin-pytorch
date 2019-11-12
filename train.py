import argparse
import datetime
import logging
import os.path
import sys
import time

import common
import cv2
import datasets.utils
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


# standard training params
add_arg("--cpuonly", action="store_true")
add_arg("--num_epochs", type=int, default=10)
add_arg("--num_steps", type=int, default=100000)
add_arg("--test_freq", type=int, default=5000)
add_arg("--test_every_epoch", action="store_true")
add_arg("--print_freq", type=int, default=10)
add_arg("--save_freq", type=int, default=5000)
add_arg("--bs", type=int, default=1024)
add_arg("--optim", type=str, default="sgd")
add_arg("--lr", type=float, default=0.1)
add_arg("--lr_policy", type=str, default="linear")
add_arg("--momentum", type=float, default=0.9)
add_arg("--dampening", type=float, default=0)
add_arg("--wd", type=float, default=0.0001)
add_arg("--pretrained", type=str, default=None)
add_arg("--output_root", type=str, default="./trained")
add_arg("--suffix", type=str, default="")

# loss params
add_arg("--margin", type=float, default=1.0)
add_arg("--loss_type", type=str, default="dsm")
add_arg("--binary", action="store_true")
add_arg("--no_data_aug", action="store_true")

# data params
add_arg("--train_data", nargs="+", type=str, default=["brown.liberty"])
add_arg("--test_data", nargs="+", type=str, default=[])
add_arg("--patch_size", type=int, default=32)

args = parser.parse_args()

# select device
device = "cpu" if args.cpuonly else "cuda"

# set up the model
model = L2Net(out_dim=256 if args.binary else 128, binary=args.binary)
if args.pretrained is not None:
    model.load_state_dict(torch.load(args.pretrained))

model = model.to(device)
if not args.cpuonly and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = common.create_optimizer(
    optimizer_type=args.optim,
    model_params=model.parameters(),
    lr=args.lr,
    wd=args.wd,
    momentum=args.momentum,
    dampening=args.dampening,
)

# set up criterion
if args.loss_type == "hardnet":
    criterion = HardNetLoss(args.margin, is_binary=args.binary)
elif args.loss_type == "dsm":
    criterion = DynamicSoftMarginLoss(is_binary=args.binary, nbins=args.bs // 2)
else:
    raise ValueError(f"{args.loss_type} is an unknown loss type!")
criterion = criterion.to(device)

# setup training and validation data
assert args.train_data, misc.yellow("training data is empty")

dset_means = []
dset_stds = []
dset_lengths = []

for dset_str in args.train_data:
    parts = dset_str.split(".")
    dset_name, seq_name = dset_str.split(".")
    if dset_name == "brown":
        dset = BrownDataset
    elif dset_name == "HP":
        dset = HPatches
    else:
        raise ValueError(f"{dset_name} is not a test dataset")

    dset_means.append(dset.mean[seq_name])
    dset_stds.append(dset.std[seq_name])
    dset_lengths.append(dset.length[seq_name])

avg_mean, avg_std = common.compute_multi_dataset_mean_std(
    dset_lengths, dset_means, dset_stds
)
tforms = common.get_basic_input_transform(args.patch_size, avg_mean, avg_std)

train_dsets = []
for dset_str in args.train_data:
    dset_name, seq_name = dset_str.split(".")
    if dset_name == "HP":
        dset = HPatches(
            root="./data/hpatches",
            num_types=16,
            transform=tforms,
            data_aug=not args.no_data_aug,
            download=True,
        )
    elif dset_name == "brown":
        dset = BrownDataset(
            root="./data/brown",
            name=seq_name,
            download=True,
            train=True,
            transform=tforms,
            triplet=False,
            data_aug=not args.no_data_aug,
        )
    else:
        raise ValueError("dataset not recognized")
    train_dsets.append(dset)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset(train_dsets),
    batch_size=args.bs,
    shuffle=True,
    num_workers=4,
    drop_last=True,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed(np.random.get_state()[1][0] + x),
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
    if dset_name == "HP":
        raise ValueError(misc.yellow("HPatches should be evaluated offline"))
    elif dset_name == "brown":
        dset = BrownDataset(
            root="./data/brown",
            name=seq_name,
            download=True,
            train=False,
            transform=tforms,
            data_aug=False,
        )
    else:
        raise ValueError("dataset not recognized")

    logger.info(f"adding evaluator {dset_dot_seq}")
    evaluators[dset_dot_seq] = DescriptorEvaluator(
        extractor=desc_extractor,
        datasets=dset,
        batch_size=args.bs,
        binarize=args.binary,
    )

# scheduler
# overwrites args.num_epochs is num_steps is specified
if args.num_steps is not None:
    args.num_epochs = (args.num_steps + len(train_loader) - 1) // len(train_loader)
args.num_steps = args.num_epochs * len(train_loader)

lr_scheduler = common.get_lr_scheduler(
    optimizer, lr_policy=args.lr_policy, num_steps=args.num_steps
)

# make output folder
s = "TRAIN"
if len(args.train_data) < 3:
    for t in args.train_data:
        s += "_" + t
else:
    s += "_" + f"{len(args.train_data)}dsets"
s += "_" + args.loss_type
s += "_binary" if args.binary else ""
s += "_" + args.suffix if args.suffix != "" else ""
save_dir = f"{args.output_root}/{s}"

logger.info(misc.green("result path = " + save_dir))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# save arguments
with open(os.path.join(save_dir, "args.txt"), "w") as f:
    f.write(" ".join(sys.argv))
    f.close()

# save mean std
torch.save({"mean": avg_mean, "std": avg_std}, os.path.join(save_dir, "mean_std.pt"))

# overwrites args.test_freq if args.test_every_epoch is set
if args.test_every_epoch:
    args.test_freq = len(train_loader)


def test():
    logger.info("running evaluation...")
    fpr95 = {}
    for dset_dot_seq, evaluator in evaluators.items():
        evaluator.run()
        fpr95[dset_dot_seq] = evaluator.computeFPR95()
    return fpr95


def train(epochs=None, steps=None):
    for epoch_idx in range(epochs):
        np.random.seed()
        for batch_idx, batch_data in enumerate(train_loader):
            step = epoch_idx * len(train_loader) + batch_idx
            if step >= steps:  # stop
                break

            # logging
            scalar_log = {}
            image_log = {}

            data = torch.cat(batch_data, dim=0)

            # forward-loss-backprop
            optimizer.zero_grad()
            out = model(data.to(device))
            loss = criterion(out)
            loss.backward()
            optimizer.step()

            # change learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

            if (step + 1) % args.print_freq == 0:
                logger.info(
                    f"Epoch {epoch_idx} ({batch_idx}/{len(train_loader)}) | Loss={loss.item():.3f}"
                )

            if (step + 1) % args.test_freq == 0:
                test_result = test()
                for dset_dot_seq, fpr95 in test_result.items():
                    logger.info(f"FPR95-{dset_dot_seq} = {fpr95 * 100}%")
                model.train()

            # save models
            if (step + 1) % args.save_freq == 0:
                if isinstance(model, nn.DataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                torch.save(state_dict, f"{save_dir}/model.state_dict")


# begin training
start_train_t = time.time()
train(args.num_epochs, args.num_steps)
hours_took = (time.time() - start_train_t) / 3600.0
logger.info(f"training took {hours_took:.2f} hours")
