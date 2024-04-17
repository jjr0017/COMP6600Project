import sys
import os
import sklearn
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import numpy as np

def get_confusion_matrix(y_truth, y_predicted):
    return sklearn.metrics.confusion_matrix(y_truth, y_predicted)

def get_classification_report(y_truth, y_predicted):
    return sklearn.metrics.classification_report(y_truth, y_predicted)

def evaluate_model(model_name, y_truth, y_predicted, log_loss=None):
    confusion_matrix = get_confusion_matrix(y_truth, y_predicted)
    classification_report = get_classification_report(y_truth, y_predicted)
    print(model_name)
    print(confusion_matrix)
    print(classification_report)
    print()

    if log_loss is not None:
        plt.plot(log_loss[0, :], log_loss[1, :])
        plt.xlabel('Number of Samples Trained')
        plt.ylabel('Log Loss')
        plt.savefig(model_name+'_loss_curve.png')
        plt.clf()

def plot_model_learning_curves(models, x, y):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

    common_params = {
        "X": x,
        "y": y,
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    for ax_idx, estimator in enumerate(models):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")