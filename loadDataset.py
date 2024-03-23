import datasets
import pandas as pd

dataset_info = datasets.load_dataset_builder("tpremoli/CelebA-attrs")

print(dataset_info.info.features)
print(dataset_info.info.splits)

dataset = datasets.load_dataset("tpremoli/CelebA-attrs", split="train")
