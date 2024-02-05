"""Down stream task evaluation."""
import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from scipy.stats import percentileofscore
from sklearn.metrics import auc, roc_curve


def corner_plot(data, true_value=None):
    if true_value is None:
        return corner(
            data,
            levels=(0.5, 0.9),
            scale_hist=True,
        )
    return corner(
        data,
        levels=(0.5, 0.9),
        scale_hist=True,
        labels=true_value.keys(),
        truths=list(true_value.values()),
    )


def roc_plot(label, pred):
    fpr, tpr, _ = roc_curve(label, pred)
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (area = {:.2f})".format(area))
    plt.legend(loc=2)
    plt.show()


def pp_plot(label, pred):
    """pp plot of the posterior probabilities
    args:
    label: true label shape (n_events, n_dims)
    pred: predicted label shape (n_events, n_samples, n_dims)
    """
    percentiles = np.empty((label.shape[0], label.shape[2]))
    for idx in range(label.shape[0]):
        for n in range(label.shape[2]):
            percentiles[idx, n] = percentileofscore(pred[idx, :, n], label[idx:n])
