import pickle
import os
import torchtext
import torch
import data_hyperparameters
from datetime import datetime

VOCAB_SIZE = data_hyperparameters.VOCAB_SIZE
TOKENIZER = data_hyperparameters.TOKENIZER

VOCAB_FILE = f"imdb_vocab_{TOKENIZER}_{VOCAB_SIZE}.pkl"
tokenizer = torchtext.data.utils.get_tokenizer(TOKENIZER)


def get_vocab():
    if not os.path.exists(VOCAB_FILE):
        dataset_fit_raw, dataset_test_raw = torchtext.experimental.datasets.IMDB(tokenizer=tokenizer)
        assert len(dataset_fit_raw) == 25000
        assert len(dataset_test_raw) == 25000
        vocab = dataset_fit_raw.get_vocab()
        new_vocab = torchtext.vocab.Vocab(counter=vocab.freqs, max_size=VOCAB_SIZE)
        save_vocab(new_vocab, VOCAB_FILE)
    else:
        new_vocab = pickle.load(open(VOCAB_FILE, "rb"))
    return new_vocab


def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


def get_data():
    vocab = get_vocab()
    dataset_fit, dataset_test = torchtext.experimental.datasets.IMDB(tokenizer=tokenizer, vocab=vocab)
    return vocab, dataset_fit, dataset_test


def get_bow(sentence, vocab_size):
    vector = [0.] * vocab_size
    for word in sentence:
        vector[word] += 1 / len(sentence)
    return vector


def get_bow_tensor(dataset, vocab_size):
    xs = torch.tensor([get_bow(x, vocab_size) for _, x in dataset], dtype=torch.float)
    return xs


def get_y_tensor(dataset):
    ys = torch.tensor([y for y, _ in dataset], dtype=torch.long)
    return ys


def augment_dataset(dataset):
    samples = [(len(txt), idx, label, txt) for idx, (label, txt) in enumerate(dataset)]
    samples.sort()  # sort by length and pad sequences with similar lengths
    return samples


def pad_batch(batch, PAD_TOKEN):
    # Find max length of the batch
    max_len = max([sample[0] for sample in batch])
    ys = torch.tensor([sample[2] for sample in batch], dtype=torch.long)
    xs = [sample[3] for sample in batch]
    xs_padded = torch.stack([torch.cat((x, torch.tensor([PAD_TOKEN] * (max_len - len(x))).long())) for x in xs])
    return xs_padded, ys


loader_fit = torch.utils.data.DataLoader(dataset=augment_dataset(dataset_fit), batch_size=8, collate_fn=pad_batch)
loader_val = torch.utils.data.DataLoader(dataset=augment_dataset(dataset_val), batch_size=8, collate_fn=pad_batch)

