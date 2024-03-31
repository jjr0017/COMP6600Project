import sys
from sklearn import tree

import argParser
import loadDataset

models, clear = argParser.parse_args(sys.argv)

if clear:
    loadDataset.clear_cache()
    print("Cleared huggingface cache")
    exit()

celeb_training_dataset, celeb_training_targets = loadDataset.load_training_dataset()
celeb_testing_dataset, celeb_testing_targets = loadDataset.load_testing_dataset()

# Train Models
dt = tree.DecisionTreeClassifier()
dt.fit(celeb_training_dataset, celeb_training_targets)
tree.plot_tree(dt)


# Validation of Models



# Metrics