## Learning Local Descriptors with a CDF-Based Dynamic Soft Margin

This is the official PyTorch implementation of the ICCV'19 oral paper *Learning Local Descriptors with a CDF-Based Dynamic Soft Margin* by Linguang Zhang and Szymon Rusinkiewicz.

### Dependencies
- Python 3.7 (we recommend using `conda` for package management)
- PyTorch 1.3
- Python-OpenCV
- Scipy

With `conda`, dependencies can be installed by:
```sh 
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge opencv
conda install -c anaconda scipy
```
NOTE: The code has been tested with PyTorch 1.3.

### Dataset
Both the PhotoTourim datasets and HPatches datasets will be automatically downloaded when the training/testing script runs. By default they will be downloaded to `./data/`. We use *dot* to separate the dataset name and the split name. PhotoTourism (Brown) datasets are named as `"brown.liberty"`, `"brown.yosemite"` and `"brown.notredame"` respectively. We use the full HPatches dataset during training, which is named as `"HP.full"`.

### Training
Use the following command to train a model with the `liberty` sequence of PhotoTourism.
```sh
CUDA_VISIBLE_DEVICES=0,1 python train.py --train_data brown.liberty
```
NOTE: we use two GPUs for our experiments. One GPU is typically fine for most modern setups.


### Testing
Use the following command to test a real-valued descriptor model:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --test_data brown.yosemite brown.notredame --model_dir pretrained/liberty_float
```
To test a binary descriptor model, add the `--binary` flag:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --test_data brown.yosemite brown.notredame --model_dir pretrained/liberty_binary --binary
```

#### Note on Pretrained Models
Our models have been retrained with the latest PyTorch release after the conference. Therefore the current evaluation results subtly differ from the numbers presented in the paper.

### Citing
If you find our code or models useful in your research, please consider citing:
```
@InProceedings{Zhang_2019_ICCV,
    author = {Zhang, Linguang and Rusinkiewicz, Szymon},
    title = {Learning Local Descriptors With a CDF-Based Dynamic Soft Margin},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

This code was initially inspired by the HardNet implementation, please consider citing the HardNet paper as well:
```
@article{HardNet2017,
    author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins: Local descriptor learning loss}",
    booktitle = {Proceedings of NIPS},
    year = 2017,
    month = dec
}
```