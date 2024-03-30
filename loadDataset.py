import datasets
import pandas as pd


def load_dataset():
    dataset_info = datasets.load_dataset_builder("tpremoli/CelebA-attrs")

    print(dataset_info.info.features)
    print(dataset_info.info.splits)

    celeb_dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="train")

    return celeb_dataset

if __name__ == '__main__':
    celeb_dataset = load_dataset()
