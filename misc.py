import torch
from collections import OrderedDict


def red(str):
    return '\33[31m' + str + '\033[0m'


def green(str):
    return '\33[32m' + str + '\033[0m'


def yellow(str):
    return '\33[33m' + str + '\033[0m'


def blue(str):
    return '\33[34m' + str + '\033[0m'


def torch_load_state_dict(state_dict_path):
    state_dict = torch.load(state_dict_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module'):
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    # load params
    return new_state_dict
