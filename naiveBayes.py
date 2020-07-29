from torchtext.experimental.datasets import IMDB
from torchtext.vocab import Vocab
import torchtext
import os
import pickle
import numpy as np

VOCAB_SIZE = 10000
TOKENIZER = "spacy"

def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

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

VOCAB_FILE = f"imdb_vocab_{TOKENIZER}_{VOCAB_SIZE}.pkl"
tokenizer = torchtext.data.utils.get_tokenizer(TOKENIZER)

if not os.path.exists(VOCAB_FILE):
    dataset_fit_raw, dataset_test_raw = IMDB(tokenizer=tokenizer)
    assert len(dataset_fit_raw) == 25000
    assert len(dataset_test_raw) == 25000
    vocab = dataset_fit_raw.get_vocab()
    print(f"Original vocab size: {len(vocab)}")
    new_vocab = Vocab(counter=vocab.freqs, max_size=VOCAB_SIZE)
    save_vocab(new_vocab, VOCAB_FILE)
else:
    new_vocab = pickle.load(open(VOCAB_FILE, "rb"))

dataset_fit, dataset_test = IMDB(tokenizer=tokenizer, vocab=new_vocab)

# First position for the number of times word has been positive.
# Second position for the total number of times seen
scores = np.zeros((len(new_vocab), 2))
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
