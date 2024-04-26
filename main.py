import sys
import numpy as np
from tqdm import tqdm
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import argParser
import loadDataset
import modelEvaluation

INCREMENT = 1000

models, clear, gen_loss_curve = argParser.parse_args(sys.argv)

models_used = {}

if clear:
    loadDataset.clear_cache()
    print("Cleared huggingface cache")
    exit()

celeb_training_dataset = loadDataset.load_training_dataset()
celeb_testing_dataset = loadDataset.load_testing_dataset()
celeb_validation_dataset = loadDataset.load_validation_dataset()

# baseline model
x, y = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
y_predicted = [None for _ in range(len(y))]
for i in range(len(y)):
    rand = np.random.randint(1, 5)
    if rand == 1:
        y_predicted[i] = 'Black_Hair'
    elif rand == 2:
        y_predicted[i] = 'Blond_Hair'
    elif rand == 3:
        y_predicted[i] = 'Brown_Hair'
    elif rand == 4:
        y_predicted[i] = 'Gray_Hair'

modelEvaluation.evaluate_model('baseline', y, y_predicted)


# Train Models
if models['dt']:
    print("Training Decision Tree model...")
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2)
    x, y = loadDataset.configure_dataset(celeb_training_dataset['image'], celeb_training_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    dt.fit(x, y)
    x, y = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    y_predicted = dt.predict(x)
    modelEvaluation.evaluate_model('dt', y, y_predicted)
    models_used['dt'] = dt

if models['nb']:
    print("Training Naive Bayes model...")
    nb = CategoricalNB(min_categories=256, class_prior=None)
    loss = None
    if gen_loss_curve:
        loss = np.zeros((2, INCREMENT))
    x_test, y_test = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    samples = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        # print(loadDataset.count_occurences(y, 'None'), '/', len(y), 'are not classified')
        nb.partial_fit(x, y, ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
        samples += len(x)
        if gen_loss_curve:
            y_predicted_loss = nb.predict(x_test)
            loss[0, idx] = samples
            loss[1, idx] = sklearn.metrics.log_loss(y_test, loadDataset.labels_to_matrix(y_predicted_loss))
    y_predicted = nb.predict(x_test)
    modelEvaluation.evaluate_model('nb', y_test, y_predicted, loss)
    models_used['nb'] = nb

if models['lr']:
    print("Training Linear Regression model...")
    lr = SGDClassifier(max_iter = 1000, tol = 1e-3, penalty = None, eta0 = 0.1, loss='hinge')
    loss = None
    if gen_loss_curve:
        loss = np.zeros((2, INCREMENT))
    x_test, y_test = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    samples = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        # print(loadDataset.count_occurences(y, 'None'), '/', len(y), 'are not classified')
        lr.partial_fit(x, y, ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
        samples += len(x)
        if gen_loss_curve:
            y_predicted_loss = lr.predict(x_test)
            loss[0, idx] = samples
            loss[1, idx] = sklearn.metrics.log_loss(y_test, loadDataset.labels_to_matrix(y_predicted_loss))
    y_predicted = lr.predict(x_test)
    modelEvaluation.evaluate_model('lr', y_test, y_predicted, loss)
    models_used['lr'] = lr

if models['svm']:
    print("Training Support Vector Machine model...")
    svm = SVC(max_iter=100000, tol=0.0001)
    x, y = loadDataset.configure_dataset(celeb_training_dataset['image'], celeb_training_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    svm.fit(x, y)
    x, y = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    y_predicted = svm.predict(x)
    modelEvaluation.evaluate_model('svm', y, y_predicted)
    models_used['svm'] = svm

if models['mlp']:
    print("Training Multi-layered Perceptron model...")
    mlp = MLPClassifier(hidden_layer_sizes=[1000, 1000, 1000, 1000], activation='relu', max_iter=1000, tol=0.00001, alpha=0.001)
    loss = None
    if gen_loss_curve:
        loss = np.zeros((2, INCREMENT))
    x_test, y_test = loadDataset.configure_dataset(celeb_testing_dataset['image'], celeb_testing_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
    samples = 0
    for idx in tqdm(range(INCREMENT)):
        x, y = loadDataset.get_data_shard_as_np_array(celeb_training_dataset, INCREMENT, idx)
        # print(loadDataset.count_occurences(y, 'None'), '/', len(y), 'are not classified')
        mlp.partial_fit(x, y, ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'])
        samples += len(x)
        if gen_loss_curve:
            y_predicted_loss = mlp.predict(x_test)
            loss[0, idx] = samples
            loss[1, idx] = sklearn.metrics.log_loss(y_test, loadDataset.labels_to_matrix(y_predicted_loss))
    y_predicted = mlp.predict(x_test)
    modelEvaluation.evaluate_model('mlp', y_test, y_predicted, loss)
    models_used['mlp'] = mlp

# Validation of Models
print("Model Validation")
x_validate, y_validate = loadDataset.configure_dataset(celeb_validation_dataset['image'], celeb_validation_dataset.select_columns(['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']))
for m in models_used.keys():
    y_predicted_validation = models_used[m].predict(x_validate)
    modelEvaluation.evaluate_model(m+'_validation', y_validate, y_predicted_validation)
