import sys

import argParser
import loadDataset

models, clear = argParser.parse_args(sys.argv)

if clear:
    loadDataset.clear_cache()
    print("Cleared huggingface cache")
    exit()

celeb_training_dataset = loadDataset.load_training_dataset()
celeb_testing_dataset = loadDataset.load_testing_dataset()

# Train Models



# Validation of Models



# Metrics