import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
if models['dt']:
    dt = DecisionTreeClassifier()
    dt.fit(celeb_training_dataset, celeb_training_targets)
    # tree.plot_tree(dt)

if models['nb']:
   nb = GaussianNB()
   nb.fit(celeb_training_dataset, celeb_training_targets) 

if models['lr']:
    lr = LinearRegression()
    lr.fit(celeb_training_dataset, celeb_training_targets)

if models['svm']:
    svm = SVC()
    svm.fit(celeb_training_dataset, celeb_training_targets)

if models['mlp']:
    mlp = MLPClassifier()
    mlp.fit(celeb_training_dataset, celeb_training_targets)

# Validation of Models



# Metrics