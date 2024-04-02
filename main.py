import sys
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import argParser
import loadDataset

INCREMENT = 1000

models, clear = argParser.parse_args(sys.argv)

if clear:
    loadDataset.clear_cache()
    print("Cleared huggingface cache")
    exit()

celeb_training_dataset = loadDataset.load_training_dataset()
celeb_testing_dataset = loadDataset.load_testing_dataset()

# Train Models
if models['dt']:
    dt = DecisionTreeClassifier()

    idx = 0
    for _ in tqdm(range(0, len(celeb_training_dataset), INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        dt.fit(x, y)
        idx += 1
        if idx > 5:
            break
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = dt.predict(x)
    print(y_predicted)
    # tree.plot_tree(dt)

# if models['nb']:
#    nb = GaussianNB()
#    nb.fit(celeb_training_dataset, celeb_training_targets) 

# if models['lr']:
#     lr = LinearRegression()
#     lr.fit(celeb_training_dataset, celeb_training_targets)

# if models['svm']:
#     svm = SVC()
#     svm.fit(celeb_training_dataset, celeb_training_targets)

# if models['mlp']:
#     mlp = MLPClassifier()
#     mlp.fit(celeb_training_dataset, celeb_training_targets)

# Validation of Models



# Metrics