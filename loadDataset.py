import datasets
import numpy as np
from tqdm import tqdm
import pickle

import os
import shutil

def configure_dataset(celeb_dataset):
    X = np.zeros((len(celeb_dataset), len(np.array(celeb_dataset[0]['image']).flatten())))
    y = np.zeros((len(celeb_dataset), 4))
    for i in tqdm(range(len(celeb_dataset))):
        X[i] = np.array(celeb_dataset[0]['image']).flatten()
        if celeb_dataset[i]['Black_Hair'] == 1:
            y[i][0] = 1
        if celeb_dataset[i]['Blond_Hair'] == 1:
            y[i][1] = 1
        if celeb_dataset[i]['Brown_Hair'] == 1:
            y[i][2] = 1
        if celeb_dataset[i]['Gray_Hair'] == 1:
            y[i][3] = 1

    return X, y

def load_training_dataset():
    print('Loading training dataset... ')
    dataset_pickle_filename = 'training_dataset.pickle'
    target_pickle_filename = 'training_targets.pickle'

    celeb_dataset = None
    targets = None
    if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        with open(dataset_pickle_filename, 'rb') as handle:
            celeb_dataset = pickle.load(handle)
        with open(target_pickle_filename, 'rb') as handle:
            targets = pickle.load(handle)
    else:
        celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="train")

        celeb_dataset, targets = configure_dataset(celeb_dataset)

        with open(dataset_pickle_filename, 'wb') as handle:
            pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(target_pickle_filename, 'wb') as handle:
            pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

    return celeb_dataset, targets

def load_testing_dataset():
    print('Loading testing dataset... ')
    dataset_pickle_filename = 'testing_dataset.pickle'
    target_pickle_filename = 'testing_targets.pickle'

    celeb_dataset = None
    targets = None
    if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        with open(dataset_pickle_filename, 'rb') as handle:
            celeb_dataset = pickle.load(handle)
        with open(target_pickle_filename, 'rb') as handle:
            targets = pickle.load(handle)
    
    else:
        celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="test")
        celeb_dataset, targets = configure_dataset(celeb_dataset)

        with open(dataset_pickle_filename, 'wb') as handle:
            pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(target_pickle_filename, 'wb') as handle:
            pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    return celeb_dataset, targets

def load_validation_dataset():
    print('Loading validation dataset... ')
    dataset_pickle_filename = 'validation_dataset.pickle'
    target_pickle_filename = 'validation_targets.pickle'

    celeb_dataset = None
    targets = None
    if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        with open(dataset_pickle_filename, 'rb') as handle:
            celeb_dataset = pickle.load(handle)
        with open(target_pickle_filename, 'rb') as handle:
            targets = pickle.load(handle)
    
    else:
        celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="validation")
        celeb_dataset, targets = configure_dataset(celeb_dataset)

        with open(dataset_pickle_filename, 'wb') as handle:
            pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(target_pickle_filename, 'wb') as handle:
            pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done.")
    return celeb_dataset, targets

def clear_cache():
    shutil.rmtree(os.path.expanduser('~/.cache/huggingface/'))
    os.remove('training_dataset.pickle')
    os.remove('training_targets.pickle')
    os.remove('testing_dataset.pickle')
    os.remove('testing_targets.pickle')
    os.remove('validation_dataset.pickle')
    os.remove('validation_targets.pickle')


if __name__ == '__main__':
    celeb_dataset = load_training_dataset()
    celeb_dataset = load_testing_dataset()
    celeb_dataset = load_validation_dataset()
