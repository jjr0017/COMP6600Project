import datasets
import numpy as np
from tqdm import tqdm
import pickle

import os
import shutil

NUM_TRAINING_SAMPLES = 10000
NUM_TESTING_SAMPLES = 2000
NUM_VALIDATION_SAMPLES = 2000

def labels_to_matrix(y):
    y_mat = np.zeros((len(y), 4))
    for i, y in enumerate(y):
        if y == 'Black_Hair':
            y_mat[i, 0] = 1
        elif y == 'Blond_Hair':
            y_mat[i, 1] = 1
        elif y == 'Brown_Hair':
            y_mat[i, 2] = 1
        elif y == 'Gray_Hair':
            y_mat[i, 3] = 1
    
    return y_mat

def count_occurences(l, element):
    count = 0
    for ele in l:
        if ele == element:
            count += 1

    return count

def clean_dataset(x, y):
    y = np.array(y)
    if len(x) != len(y):
        raise("x not equal to y in clean_dataset")
    
    idx_mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if np.max(x[i]) > 255 or np.min(x[i]) < 0:
            idx_mask[i] = False
        
        if y[i] == 'None':
            idx_mask[i] = False
    
    x = x[idx_mask, ...]
    y = y[idx_mask, ...]

    return x, y

def configure_dataset(ds, targets):
    x = np.zeros((len(ds), 178*218*3))
    for i in range(len(ds)):
        x[i] = np.array(ds[i]).flatten()
    y = ['None' for _ in range(len(targets))]
    for i in range(len(targets)):
        if int(targets['Black_Hair'][i]) == 1:
            y[i] = 'Black_Hair'
        if int(targets['Blond_Hair'][i]) == 1:
            y[i] = 'Blond_Hair'
        if int(targets['Brown_Hair'][i]) == 1:
            y[i] = 'Brown_Hair'
        if int(targets['Gray_Hair'][i]) == 1:
            y[i] = 'Gray_Hair'

    x, y = clean_dataset(x, y)
    return x, y

def get_data_shard_as_np_array(ds, num_shards, index):

    training_dataset =ds.shard(num_shards=num_shards, index=index)['image']

    training_targets = ds.shard(num_shards=num_shards, index=index).select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    x, y = configure_dataset(training_dataset, training_targets)
    return x, y

# def configure_dataset(celeb_dataset):
#     # X = np.zeros((len(celeb_dataset), len(np.array(celeb_dataset[0]['image']).flatten())))
#     # y = np.zeros((len(celeb_dataset), 4))
#     # for i in tqdm(range(len(celeb_dataset))):
#     #     # X[i] = np.array(celeb_dataset[0]['image']).flatten()
#     #     if celeb_dataset[i]['Black_Hair'] == 1:
#     #         y[i][0] = 1
#     #     if celeb_dataset[i]['Blond_Hair'] == 1:
#     #         y[i][1] = 1
#     #     if celeb_dataset[i]['Brown_Hair'] == 1:
#     #         y[i][2] = 1
#     #     if celeb_dataset[i]['Gray_Hair'] == 1:
#     #         y[i][3] = 1

#     # return X, y
#     # print(celeb_dataset)
#     # celeb_dataset = celeb_dataset.select_columns(['image', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
#     # celeb_dataset['image']
#     return celeb_dataset

def load_training_dataset():
    print('Loading training dataset... ')
    # dataset_pickle_filename = 'training_dataset.pickle'
    # target_pickle_filename = 'training_targets.pickle'

    # celeb_dataset = None
    # targets = None
    # if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        # with open(dataset_pickle_filename, 'rb') as handle:
            # celeb_dataset = pickle.load(handle)
        # with open(target_pickle_filename, 'rb') as handle:
            # targets = pickle.load(handle)
    # else:
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="train")
    # idx_with_data = np.zeros(len(celeb_dataset))
    # for row in range(len(celeb_dataset)):
    #     if row % 1000 == 0:
    #         print(row, "/", len(celeb_dataset))
    #     if int(celeb_dataset['Black_Hair'][row]) == 1 or int(celeb_dataset['Blond_Hair'][row]) == 1 or int(celeb_dataset['Brown_Hair'][row]) == 1 or int(celeb_dataset['Gray_Hair'][row]) == 1:
    #         idx_with_data[row] = 1
    # print(np.sum(idx_with_data))
    if NUM_TRAINING_SAMPLES is not None:
        celeb_dataset = celeb_dataset.shuffle(seed=1856).select(range(NUM_TRAINING_SAMPLES))

    # celeb_dataset = configure_dataset(celeb_dataset)

        # with open(dataset_pickle_filename, 'wb') as handle:
            # pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(target_pickle_filename, 'wb') as handle:
            # pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

    return celeb_dataset

def load_testing_dataset():
    print('Loading testing dataset... ')
    # dataset_pickle_filename = 'training_dataset.pickle'
    # target_pickle_filename = 'training_targets.pickle'

    # celeb_dataset = None
    # targets = None
    # if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        # with open(dataset_pickle_filename, 'rb') as handle:
            # celeb_dataset = pickle.load(handle)
        # with open(target_pickle_filename, 'rb') as handle:
            # targets = pickle.load(handle)
    # else:
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="test")
    if NUM_TESTING_SAMPLES is not None:
        celeb_dataset = celeb_dataset.shuffle(seed=1856).select(range(NUM_TESTING_SAMPLES))

    # celeb_dataset = configure_dataset(celeb_dataset)

        # with open(dataset_pickle_filename, 'wb') as handle:
            # pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(target_pickle_filename, 'wb') as handle:
            # pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

    return celeb_dataset

def load_validation_dataset():
    print('Loading validation dataset... ')
    # dataset_pickle_filename = 'training_dataset.pickle'
    # target_pickle_filename = 'training_targets.pickle'

    # celeb_dataset = None
    # targets = None
    # if os.path.isfile(dataset_pickle_filename) and os.path.isfile(target_pickle_filename):

        # with open(dataset_pickle_filename, 'rb') as handle:
            # celeb_dataset = pickle.load(handle)
        # with open(target_pickle_filename, 'rb') as handle:
            # targets = pickle.load(handle)
    # else:
    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="validation")
    if NUM_VALIDATION_SAMPLES is not None:
        celeb_dataset = celeb_dataset.shuffle(seed=1856).select(range(NUM_VALIDATION_SAMPLES))

    # celeb_dataset = configure_dataset(celeb_dataset)

        # with open(dataset_pickle_filename, 'wb') as handle:
            # pickle.dump(celeb_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(target_pickle_filename, 'wb') as handle:
            # pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')

    return celeb_dataset

def clear_cache():
    shutil.rmtree(os.path.expanduser('~/.cache/huggingface/'))
    # os.remove('training_dataset.pickle')
    # os.remove('training_targets.pickle')
    # os.remove('testing_dataset.pickle')
    # os.remove('testing_targets.pickle')
    # os.remove('validation_dataset.pickle')
    # os.remove('validation_targets.pickle')


if __name__ == '__main__':
    celeb_dataset = load_training_dataset()
    celeb_dataset = load_testing_dataset()
    celeb_dataset = load_validation_dataset()
