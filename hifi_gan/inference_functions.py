from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from .env import AttrDict
from .meldataset import MAX_WAV_VALUE
from .models import Generator


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def load_generator(checkpoint_file, device="cuda" if torch.cuda.is_available() else "cpu"):
    # If checkpoint file is absolute, use that, otherwise use relative path to this file
    if not os.path.isabs(checkpoint_file):
        relative_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint_file = os.path.join(relative_path, checkpoint_file)

    config_file = os.path.join(os.path.split(checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    config = AttrDict(json_config)

    torch.manual_seed(config.seed)
    generator = Generator(config).to(device)

    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()

    return generator


def run_hifigan_inference(generator, mel):
    with torch.no_grad():
        x = torch.FloatTensor(mel)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype("int16")

    return audio
