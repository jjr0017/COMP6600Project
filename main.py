import sys
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
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
    print("Training Decision Tree model...")
    dt = DecisionTreeClassifier()
    # dt = RandomForestClassifier(warm_start=True, n_estimators=1000)

    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        dt.fit(x, y)
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = dt.predict(x)
    print(y_predicted)

if models['nb']:
    print("Training Naive Bayes model...")
    nb = CategoricalNB()
    idx = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        nb.partial_fit(x, y, ['None', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = nb.predict(x)
    print(y_predicted)

if models['lr']:
    print("Training Linear Regression model...")
    lr = SGDClassifier(max_iter = 500, tol = 1e-3, penalty = None, eta0 = 0.1)
    idx = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        lr.partial_fit(x, y, ['None', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = lr.predict(x)
    print(y_predicted)

if models['svm']:
    print("Training Support Vector Machine model...")
    svm = SGDClassifier(loss='log') # should SGDClassifier be used here?
    idx = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        svm.partial_fit(x, y, ['None', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = svm.predict(x)
    print(y_predicted)

if models['mlp']:
    print("Training Multi-layered Perceptron model...")
    mlp = MLPClassifier()
    idx = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        mlp.partial_fit(x, y, ['None', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
    x, y = loadDataset.get_data_shard_as_np_array(celeb_testing_dataset, 20000, 0)
    y_predicted = mlp.predict(x)
    print(y_predicted)

# Validation of Models



# Metrics