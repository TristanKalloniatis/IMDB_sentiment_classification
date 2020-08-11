import pickle
import os
import torchtext
import torch
import data_hyperparameters
from datetime import datetime
from log_utils import create_logger, write_log
from sklearn.model_selection import train_test_split

VOCAB_SIZE = data_hyperparameters.VOCAB_SIZE
TOKENIZER = data_hyperparameters.TOKENIZER
VOCAB_FILE = f"imdb_vocab_{TOKENIZER}_{VOCAB_SIZE}.pkl"
tokenizer = torchtext.data.utils.get_tokenizer(TOKENIZER)
LOG_FILE = 'data_downloader'
logger = create_logger(LOG_FILE)
device = torch.device('cuda' if data_hyperparameters.USE_CUDA else 'cpu')

if not os.path.exists('.data/'):
    os.mkdir('.data/')


def get_vocab():
    if not os.path.exists(VOCAB_FILE):
        write_log('Downloading raw data', logger)
        now = datetime.now()
        dataset_fit_raw, dataset_test_raw = torchtext.experimental.datasets.IMDB(tokenizer=tokenizer)
        write_log('Download took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
        assert len(dataset_fit_raw) == 25000
        assert len(dataset_test_raw) == 25000
        write_log('Building vocab', logger)
        vocab = dataset_fit_raw.get_vocab()
        new_vocab = torchtext.vocab.Vocab(counter=vocab.freqs, max_size=VOCAB_SIZE)
        save_data(new_vocab, VOCAB_FILE)
        write_log('Vocab saved to {0}'.format(VOCAB_FILE), logger)
    else:
        write_log('Loading vocab from {0}'.format(VOCAB_FILE), logger)
        new_vocab = pickle.load(open(VOCAB_FILE, "rb"))
    return new_vocab


def save_data(data, path):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()


def get_data():
    vocab = get_vocab()
    write_log('Building fit and test data', logger)
    now = datetime.now()
    dataset_fit, dataset_test = torchtext.experimental.datasets.IMDB(tokenizer=tokenizer, vocab=vocab)
    write_log('Building fit and test data took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
    PAD_TOKEN = dataset_fit.get_vocab()['<pad>']
    return PAD_TOKEN, dataset_fit, dataset_test


def split_data(fit_data):
    now = datetime.now()
    write_log('Splitting fit data into training and validation sets', logger)
    X_train, X_valid, y_train, y_valid = train_test_split([data[1] for data in fit_data],
                                                          [data[0] for data in fit_data],
                                                          test_size=data_hyperparameters.TRAIN_VALID_SPLIT)
    write_log('Splitting took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
    return zip(y_train, X_train), zip(y_valid, X_valid)


def get_bow(sentence, vocab_size):
    vector = [0.] * vocab_size
    for word in sentence:
        vector[word] += 1 / len(sentence)
    return vector


def get_bow_tensor(dataset, vocab_size):
    xs = torch.tensor([get_bow(x, vocab_size) for _, x in dataset], dtype=torch.float, device=device)
    return xs


def get_y_tensor(dataset):
    ys = torch.tensor([y for y, _ in dataset], dtype=torch.long, device=device)
    return ys


def get_bow_dataset(dataset, vocab_size):  # todo: test this actually works
    return torch.utils.data.TensorDataset(get_bow_tensor(dataset, vocab_size), get_y_tensor(dataset))


def augment_dataset(dataset):
    samples = [(len(txt), idx, label, txt.to(device)) for idx, (label, txt) in enumerate(dataset)]
    samples.sort()  # sort by length to pad sequences with similar lengths
    return samples


def get_dataloaders(batch_size=data_hyperparameters.BATCH_SIZE, bow=False, pack=False):
    PAD_TOKEN, dataset_fit, dataset_test = get_data()

    def pad_batch(batch):
        # Find max length of the batch
        max_len = max([sample[0] for sample in batch])
        ys = torch.tensor([sample[2] for sample in batch], dtype=torch.long, device=device)
        xs = [sample[3] for sample in batch]
        xs_padded = torch.stack(
            [torch.cat((x, torch.tensor([PAD_TOKEN] * (max_len - len(x)), device=device).long())) for x in xs])
        return xs_padded, ys

    def pad_batch_get_length(batch):
        max_len = max([sample[0] for sample in batch])
        ys = torch.tensor([sample[2] for sample in batch], dtype=torch.long, device=device)
        xs = [sample[3] for sample in batch]
        xs_padded = torch.stack(
            [torch.cat((x, torch.tensor([PAD_TOKEN] * (max_len - len(x)), device=device).long())) for x in xs])
        x_lengths = torch.tensor([len(x) for x in xs], dtype=torch.long, device=device)
        # Sort by length
        index = torch.argsort(x_lengths, descending=True)
        xs_padded = xs_padded[index]
        ys = ys[index]
        x_lengths = x_lengths[index]
        return (xs_padded, x_lengths), ys

    dataset_train, dataset_valid = split_data(dataset_fit)
    if bow:
        vocab = len(dataset_fit.get_vocab())
        return torch.utils.data.DataLoader(dataset=get_bow_dataset(dataset_train, vocab), batch_size=batch_size,
                                           shuffle=True), \
               torch.utils.data.DataLoader(dataset=get_bow_dataset(dataset_valid, vocab), batch_size=batch_size), \
               torch.utils.data.DataLoader(dataset=get_bow_dataset(dataset_test, vocab), batch_size=batch_size)
    else:
        collate_fn = pad_batch_get_length if pack else pad_batch
        return torch.utils.data.DataLoader(dataset=augment_dataset(dataset_train), batch_size=batch_size,
                                           collate_fn=collate_fn), \
               torch.utils.data.DataLoader(dataset=augment_dataset(dataset_valid), batch_size=batch_size,
                                           collate_fn=collate_fn), \
               torch.utils.data.DataLoader(dataset=augment_dataset(dataset_test), batch_size=batch_size,
                                           collate_fn=collate_fn)
