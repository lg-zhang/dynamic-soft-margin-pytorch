# torch
import json
import os.path

# built-in
import random

import cv2

# 3rd party
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
from datasets.utils import np_flip, np_rotate
from tqdm import tqdm

from torchvision.datasets.utils import download_url


class HPatches(torch.utils.data.Dataset):
    urls = {
        "data": [
            "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-release.tar.gz",
            "hpatches-release.tar.gz",
            "0ab830d37fceb2b4c86cb1cc6cc79a61",
        ],
        "splits": [
            "https://raw.githubusercontent.com/hpatches/hpatches-benchmark/master/tasks/splits/splits.json",
            "splits.json",
        ],
    }
    patch_types = [
        "ref",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
    ]
    mean = {"full": 0.421789139509}
    std = {"full": 0.226070001721}
    length = {"full": 2211472}

    def __init__(
        self, root, num_types=16, transform=None, data_aug=False, download=False
    ):
        super().__init__()
        self.root = root
        self.split_file = os.path.join(self.root, self.urls["splits"][1])
        self.data_dir = os.path.join(self.root, "hpatches-release")
        self.data_file = os.path.join(self.root, "data.pt")

        self.num_types = min(num_types, 16)
        self.transform = transform
        self.data_aug = data_aug

        if not self.data_file:
            self._read_image_file(self.root, sequences, self.patch_types)

        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        # load the serialized data
        self.data = torch.load(self.data_file)

    def __getitem__(self, index):
        p_idx1 = index
        t_idx1 = random.randint(0, self.num_types - 1)
        patch_a = self.data[t_idx1, p_idx1, ...]

        t_idx2 = t_idx1
        while t_idx2 == t_idx1:
            t_idx2 = random.randint(0, self.num_types - 1)
        patch_p = self.data[t_idx2, p_idx1, ...]

        if self.data_aug:
            patch_a, patch_p = np_rotate([patch_a, patch_p], random.randint(0, 3))
            patch_a, patch_p = np_flip([patch_a, patch_p], random.randint(0, 2))

        if self.transform is not None:
            patch_a = self.transform(patch_a)
            patch_p = self.transform(patch_p)

        return patch_a, patch_p

    def __len__(self):
        return self.data.shape[1]

    def _read_image_file(self):
        with open(self.split_file) as f:
            splits = json.load(f)
        sequences = splits["full"]["test"]

        all_patches = []
        for t in tqdm(self.patch_types):
            for seq in tqdm(sequences):
                im_p = os.path.join(self.data_dir, seq, t + ".png")
                im = cv2.imread(im_p, 0)
                N = im.shape[0] / 65
                patches = np.split(im, N)
                all_patches += patches

        return np.array(all_patches).reshape(len(self.patch_types), -1, 65, 65)

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print("# Found cached data {}".format(self.data_file))
            return

        if not self._check_downloaded():
            # download raw data
            fpath = os.path.join(self.root, self.urls["data"][1])
            download_url(
                self.urls["data"][0],
                self.root,
                filename=self.urls["data"][1],
                md5=self.urls["data"][2],
            )

            print("# Extracting data {}\n".format(fpath))

            import tarfile

            with tarfile.open(fpath, "r:gz") as t:
                t.extractall()

            os.unlink(fpath)

            # download splits.json
            download_url(
                self.urls["splits"][0], self.root, filename=self.urls["splits"][1]
            )

        # process and save as torch files
        print("# Caching data {}".format(self.data_file))

        dataset = self._read_image_file()

        with open(self.data_file, "wb") as f:
            torch.save(dataset, f, pickle_protocol=4)


def export_matfiles(root, sequences, types, save_size=32):
    for t in tqdm(types):
        for seq in tqdm(sequences):
            im_p = os.path.join(root, "data/hpatches-release", seq, t + ".png")
            im = cv2.imread(im_p, 0)
            N = im.shape[0] / 65
            patches = np.split(im, N)
            out_p = os.path.join(
                root,
                "data/hpatches-release",
                seq,
                "{}.s{}.mat".format(t, str(save_size)),
            )
            for i in range(len(patches)):
                patches[i] = cv2.resize(
                    patches[i],
                    dsize=(save_size, save_size),
                    interpolation=cv2.INTER_LINEAR,
                )
            patches = np.stack(patches)
            sio.savemat(out_p, {"patches": patches})


def remove_matfiles(root, sequences, types, save_size=32):
    for t in tqdm(types):
        for seq in tqdm(sequences):
            out_p = os.path.join(
                root,
                "data/hpatches-release",
                seq,
                "{}.s{}.mat".format(t, str(save_size)),
            )
            os.remove(out_p)


if __name__ == "__main__":
    dset = HPatches(root="./data/hpatches", download=True)
    print(len(dset))
