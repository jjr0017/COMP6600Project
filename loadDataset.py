import datasets
import pandas as pd
import numpy as np
from PIL import Image

import os
import shutil

def configure_dataset(celeb_dataset):
    print(celeb_dataset)
    print(np.array(celeb_dataset[0]['image']))

    return celeb_dataset

def load_training_dataset():
    print('Loading training dataset... ', end='', flush=True)
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="train")
    print('Done.')

    celeb_dataset = configure_dataset(celeb_dataset)

    return celeb_dataset

def load_testing_dataset():
    print('Loading testing dataset... ', end='', flush=True)
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="test")
    print('Done.')

    return celeb_dataset

def load_validation_dataset():
    print('Loading validation dataset... ', end='', flush=True)
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="validation")
    print('Done.')

    return celeb_dataset

def clear_cache():
    shutil.rmtree(os.path.expanduser('~/.cache/huggingface/'))

if __name__ == '__main__':
    celeb_dataset = load_training_dataset()
    celeb_dataset = load_testing_dataset()
    celeb_dataset = load_validation_dataset()
