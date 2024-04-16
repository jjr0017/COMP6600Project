import sys
import os
import sklearn
import sklearn.metrics

def get_confusion_matrix(y_truth, y_predicted):
    return sklearn.metrics.confusion_matrix(y_truth, y_predicted)

def get_classification_report(y_truth, y_predicted):
    return sklearn.metrics.classification_report(y_truth, y_predicted)

def evaluate_model(model_name, y_truth, y_predicted):
    confusion_matrix = get_confusion_matrix(y_truth, y_predicted)
    classification_report = get_classification_report(y_truth, y_predicted)
    print(model_name)
    print(confusion_matrix)
    print(classification_report)
    print()