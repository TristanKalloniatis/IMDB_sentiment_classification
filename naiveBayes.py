import numpy as np
import data_downloader
import data_hyperparameters
from datetime import datetime

def NB_predict(data, fitted_probs):
    words = set(data[1].tolist())
    p1 = np.prod([fitted_probs[word] for word in words])
    p0 = np.prod([1 - fitted_probs[word] for word in words])
    # No need to normalise by class probabilities as dataset is balanced
    if p0 > p1:
        return 0
    return 1

def compute_accuracy(preds, actuals):
    num_correct = 0
    for i in range(len(preds)):
        if preds[i] == actuals[i]:
            num_correct += 1
    return num_correct / len(preds)


vocab, dataset_fit, dataset_test = data_downloader.get_data()

# First position for the number of times word has been positive.
# Second position for the total number of times seen
scores = np.zeros((data_hyperparameters.VOCAB_SIZE, 2))
for data in dataset_fit:
    label = data[0].item()
    words = set(data[1].tolist())
    for word in words:
        scores[word, 1] += 1
        if label == 1:
            scores[word, 0] += 1

probs = (1 + scores[:, 0]) / (2 + scores[:, 1])  # Laplacian smoothing

dataset_fit_pred = [NB_predict(data, probs) for data in dataset_fit]
dataset_test_pred = [NB_predict(data, probs) for data in dataset_test]

fit_accuracy = compute_accuracy(dataset_fit_pred, [data[0].item() for data in dataset_fit])
test_accuracy = compute_accuracy(dataset_test_pred, [data[0].item() for data in dataset_test])
