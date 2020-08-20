import pickle
import os
import torchtext
import torch
import data_hyperparameters
from datetime import datetime
from log_utils import create_logger, write_log
from sklearn.model_selection import train_test_split
import glob
from torch.utils import data # todo: remove this

VOCAB_SIZE = data_hyperparameters.VOCAB_SIZE
TOKENIZER = data_hyperparameters.TOKENIZER
VOCAB_FILE = f"imdb_vocab_{TOKENIZER}_{VOCAB_SIZE}.pkl"
FIT_FILE = f"imdb_fit_{TOKENIZER}_{VOCAB_SIZE}.pkl"
TEST_FILE = f"imdb_test_{TOKENIZER}_{VOCAB_SIZE}.pkl"
FIT_LABELS_FILE = f"imdb_fit_labels_{TOKENIZER}_{VOCAB_SIZE}.pkl"
TEST_LABELS_FILE = f"imdb_test_labels_{TOKENIZER}_{VOCAB_SIZE}.pkl"
tokenizer = torchtext.data.utils.get_tokenizer(TOKENIZER)
LOG_FILE = 'data_downloader'
logger = create_logger(LOG_FILE)
device = torch.device('cuda' if data_hyperparameters.USE_CUDA and data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE else 'cpu')

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
    vocab_reverse_index = vocab.stoi
    PAD_TOKEN = vocab.stoi['<pad>']
    if not os.path.exists(FIT_FILE) or not os.path.exists(FIT_LABELS_FILE) or not os.path.exists(TEST_FILE) \
            or not os.path.exists(TEST_LABELS_FILE):
        write_log('Building fit and test data (this may take a while)', logger)
        now = datetime.now()
        fit_mapped_tokens = []
        fit_labels = []
        train_pos_data = glob.iglob('.data/aclImdb/train/pos/*')
        for data in train_pos_data:
            with open(data, 'r') as f:
                if data_hyperparameters.VERBOSE_LOGGING:
                    write_log('Processing {0}'.format(f), logger)
                text = f.read()
                mapped_tokens = [vocab_reverse_index[token] for token in tokenizer(text)]
            fit_mapped_tokens.append(mapped_tokens)
            fit_labels.append(1)
        train_neg_data = glob.iglob('.data/aclImdb/train/neg/*')
        for data in train_neg_data:
            with open(data, 'r') as f:
                if data_hyperparameters.VERBOSE_LOGGING:
                    write_log('Processing {0}'.format(f), logger)
                text = f.read()
                mapped_tokens = [vocab_reverse_index[token] for token in tokenizer(text)]
            fit_mapped_tokens.append(mapped_tokens)
            fit_labels.append(0)
        save_data(fit_mapped_tokens, FIT_FILE)
        save_data(fit_labels, FIT_LABELS_FILE)
        write_log('Processed fit data', logger)
        test_mapped_tokens = []
        test_labels = []
        test_pos_data = glob.iglob('.data/aclImdb/test/pos/*')
        for data in test_pos_data:
            with open(data, 'r') as f:
                if data_hyperparameters.VERBOSE_LOGGING:
                    write_log('Processing {0}'.format(f), logger)
                text = f.read()
                mapped_tokens = [vocab_reverse_index[token] for token in tokenizer(text)]
            test_mapped_tokens.append(mapped_tokens)
            test_labels.append(1)
        test_neg_data = glob.iglob('.data/aclImdb/test/neg/*')
        for data in test_neg_data:
            with open(data, 'r') as f:
                if data_hyperparameters.VERBOSE_LOGGING:
                    write_log('Processing {0}'.format(f), logger)
                text = f.read()
                mapped_tokens = [vocab_reverse_index[token] for token in tokenizer(text)]
            test_mapped_tokens.append(mapped_tokens)
            test_labels.append(0)
        save_data(test_mapped_tokens, TEST_FILE)
        save_data(test_labels, TEST_LABELS_FILE)
        write_log('Processed test data', logger)
        write_log('Building fit and test data took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
    else:
        write_log('Loading fit data from {0}'.format(FIT_FILE), logger)
        fit_mapped_tokens = pickle.load(open(FIT_FILE, "rb"))
        write_log('Loading fit labels from {0}'.format(FIT_LABELS_FILE), logger)
        fit_labels = pickle.load(open(FIT_LABELS_FILE, "rb"))
        write_log('Loading test data from {0}'.format(TEST_FILE), logger)
        test_mapped_tokens = pickle.load(open(TEST_FILE, "rb"))
        write_log('Loading test labels from {0}'.format(TEST_LABELS_FILE), logger)
        test_labels = pickle.load(open(TEST_LABELS_FILE, "rb"))
    return PAD_TOKEN, fit_mapped_tokens, fit_labels, test_mapped_tokens, test_labels


def split_data(fit_mapped_tokens, fit_labels):
    now = datetime.now()
    write_log('Splitting fit data into training and validation sets', logger)
    X_train, X_valid, y_train, y_valid = train_test_split(fit_mapped_tokens, fit_labels,
                                                          test_size=data_hyperparameters.TRAIN_VALID_SPLIT)
    write_log('Splitting took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
    return y_train, X_train, y_valid, X_valid


def augment_dataset(dataset_x, dataset_y):
    samples = [(len(txt), idx, torch.tensor(label, device=device), torch.tensor(txt, device=device))
               for idx, (label, txt) in enumerate(zip(dataset_y, dataset_x))]
    samples.sort()  # sort by length to pad sequences with similar lengths
    return samples


def get_dataloaders(batch_size=data_hyperparameters.BATCH_SIZE):
    PAD_TOKEN, fit_mapped_tokens, fit_labels, test_mapped_tokens, test_labels = get_data()

    def pad_batch(batch):
        # Find max length of the batch
        max_len = max([sample[0] for sample in batch])
        ys = torch.tensor([sample[2] for sample in batch], dtype=torch.long, device=device)
        xs = [sample[3] for sample in batch]
        xs_padded = torch.stack(
            [torch.cat((x, torch.tensor([PAD_TOKEN] * (max_len - len(x)), device=device).long())) for x in xs])
        return xs_padded, ys

    dataset_train_y, dataset_train_x, dataset_valid_y, dataset_valid_x = split_data(fit_mapped_tokens, fit_labels)

    return torch.utils.data.DataLoader(dataset=augment_dataset(dataset_train_x, dataset_train_y), batch_size=batch_size,
                                       collate_fn=pad_batch), \
           torch.utils.data.DataLoader(dataset=augment_dataset(dataset_valid_x, dataset_valid_y), batch_size=batch_size,
                                       collate_fn=pad_batch), \
           torch.utils.data.DataLoader(dataset=augment_dataset(test_mapped_tokens, test_labels), batch_size=batch_size,
                                       collate_fn=pad_batch)
