import pickle
import os
import torchtext
import torch
import glob
import data_hyperparameters
from datetime import datetime
from log_utils import create_logger, write_log
from data_downloader import get_vocab, save_data
from math import sqrt
from random import random
from torch.utils import data  # todo: remove this
import matplotlib.pyplot as plt

TOKENIZER = data_hyperparameters.TOKENIZER
VOCAB_SIZE = data_hyperparameters.VOCAB_SIZE
TOKENIZER_FILE = f"imdb_tokenized_{TOKENIZER}_{VOCAB_SIZE}.pkl"
FREQS_FILE = f"imdb_freqs_{TOKENIZER}_{VOCAB_SIZE}.pkl"
tokenizer = torchtext.data.utils.get_tokenizer(TOKENIZER)
LOG_FILE = 'embeddings'
logger = create_logger(LOG_FILE)
device = torch.device('cuda' if data_hyperparameters.USE_CUDA else 'cpu')

if not os.path.exists('.data/'):
    os.mkdir('.data/')


def get_data():
    if not os.path.exists(TOKENIZER_FILE):
        vocab = get_vocab()  # Strictly speaking this is coming from the training rather than unsupervised data
        vocab_index = vocab.itos
        vocab_reverse_index = vocab.stoi
        freqs = list(map(lambda i: vocab.freqs[vocab_index[i]], range(len(vocab))))
        total = sum(freqs)
        for i in range(len(freqs)):
            freqs[i] /= total
        save_data(freqs, FREQS_FILE)
        unsup_data = glob.iglob('.data/aclImdb/train/unsup/*')
        all_mapped_tokens = []
        write_log('Iterating through unsupervised data (this could take a while)', logger)
        now = datetime.now()
        for data in unsup_data:
            with open(data, 'r') as f:
                if data_hyperparameters.VERBOSE_LOGGING:
                    write_log('Processing {0}'.format(f), logger)
                text = f.read()
                mapped_tokens = [vocab_reverse_index[token] for token in tokenizer(text)]
            all_mapped_tokens.append(mapped_tokens)
        write_log('Iteration took {0} seconds'.format((datetime.now() - now).total_seconds()), logger)
        assert len(all_mapped_tokens) == 50000
        save_data(all_mapped_tokens, TOKENIZER_FILE)
    else:
        write_log('Loading tokens from {0}'.format(TOKENIZER_FILE), logger)
        all_mapped_tokens = pickle.load(open(TOKENIZER_FILE, "rb"))
        write_log('Loading frequencies from {0}'.format(FREQS_FILE), logger)
        freqs = pickle.load(open(FREQS_FILE, "rb"))
    return freqs, all_mapped_tokens


def subsample_probability_discard(word_frequency, threshold=data_hyperparameters.SUBSAMPLE_THRESHOLD):
    if word_frequency <= 0:
        return 0
    raw_result = 1 - sqrt(threshold / word_frequency)
    if raw_result < 0:
        return 0
    return raw_result


def subsample_word(word_frequency, threshold=data_hyperparameters.SUBSAMPLE_THRESHOLD):
    if threshold is not None:
        return random() < subsample_probability_discard(word_frequency, threshold)
    else:
        return False


def noise_distribution(frequencies, unigram_distribution_power=data_hyperparameters.UNIGRAM_DISTRIBUTION_POWER):
    adjusted_frequencies = [frequency ** unigram_distribution_power for frequency in frequencies]
    normalisation = sum(adjusted_frequencies)
    return [adjusted_frequency / normalisation for adjusted_frequency in adjusted_frequencies]


def produce_negative_samples(distribution, num_negative_samples=data_hyperparameters.NUM_NEGATIVE_SAMPLES,
                             batch_size=data_hyperparameters.WORD_EMBEDDING_BATCH_SIZE):
    return torch.multinomial(distribution, batch_size * num_negative_samples, replacement=True).view(batch_size, -1).to(device)


def pre_process_words(words, algorithm, context_size=data_hyperparameters.CONTEXT_SIZE,
                      min_review_length=data_hyperparameters.MIN_REVIEW_LENGTH):
    if len(words) < min_review_length:
        return []
    data_points = []
    for i in range(context_size, len(words) - context_size):
        context = [words[i - j - 1] for j in range(context_size)]
        context += [words[i + j + 1] for j in range(context_size)]
        target = words[i]
        if algorithm.upper() == 'CBOW':
            data_points.append((context, target))
        elif algorithm.upper() == 'SGNS':
            for word in context:
                data_points.append((word, target))
    return data_points


def build_data_loader(raw_data, frequencies, algorithm, context_size=data_hyperparameters.CONTEXT_SIZE,
                      threshold=data_hyperparameters.SUBSAMPLE_THRESHOLD,
                      min_review_length=data_hyperparameters.MIN_REVIEW_LENGTH, sub_sample=False,
                      batch_size=data_hyperparameters.WORD_EMBEDDING_BATCH_SIZE, shuffle=False):
    xs = []
    ys = []
    for review in raw_data:
        data_points = pre_process_words(review, algorithm, context_size, min_review_length)
        for data_point_x, data_point_y in data_points:
            if sub_sample:
                if subsample_word(frequencies[data_point_y], threshold):
                    continue
            xs.append(data_point_x)
            ys.append(data_point_y)
    write_log('Size of data: {0}'.format(len(xs)), logger)
    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)
    ds = torch.utils.data.TensorDataset(xs, ys)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def split_data(raw_data, train_proportion=data_hyperparameters.TRAIN_PROPORTION):
    train_data = raw_data[: int(len(raw_data) * train_proportion)]
    valid_data = raw_data[int(len(raw_data) * train_proportion):]
    return train_data, valid_data


def setup(algorithm, batch_size=data_hyperparameters.WORD_EMBEDDING_BATCH_SIZE,
          context_size=data_hyperparameters.CONTEXT_SIZE, threshold=data_hyperparameters.SUBSAMPLE_THRESHOLD,
          unigram_distribution_power=data_hyperparameters.UNIGRAM_DISTRIBUTION_POWER,
          min_review_length=data_hyperparameters.MIN_REVIEW_LENGTH):
    now = datetime.now()
    frequencies, data = get_data()
    distribution = noise_distribution(frequencies, unigram_distribution_power)
    train_data, valid_data = split_data(data)
    write_log('Train data', logger)
    train_loader = build_data_loader(train_data, frequencies, algorithm, context_size, threshold, min_review_length,
                                     sub_sample=True, batch_size=batch_size, shuffle=True)
    write_log('Validation data', logger)
    valid_loader = build_data_loader(valid_data, frequencies, algorithm, context_size, threshold, min_review_length,
                                     sub_sample=True, batch_size=2 * batch_size, shuffle=False)
    seconds = (datetime.now() - now).total_seconds()
    write_log('Setting up took: {0} seconds'.format(seconds), logger)
    return frequencies, distribution, train_loader, valid_loader

class ContinuousBagOfWords(torch.nn.Module):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE + 2,
                 embedding_dim=data_hyperparameters.WORD_EMBEDDING_DIMENSION,
                 context_size=data_hyperparameters.CONTEXT_SIZE, name='CBOW'):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.linear = torch.nn.Linear(embedding_dim, vocab_size)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.name = name
        self.algorithmType = 'CBOW'

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds.sum(dim=-2))
        log_probabilities = torch.nn.functional.log_softmax(out, dim=-1)
        return log_probabilities


class SkipGramWithNegativeSampling(torch.nn.Module):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE + 2,
                 embedding_dim=data_hyperparameters.WORD_EMBEDDING_DIMENSION,
                 context_size=data_hyperparameters.CONTEXT_SIZE,
                 num_negative_samples=data_hyperparameters.NUM_NEGATIVE_SAMPLES,
                 inner_product_clamp=data_hyperparameters.INNER_PRODUCT_CLAMP, name='SGNS'):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)  # These will be the inEmbeddings used in evaluation
        self.out_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_negative_samples = num_negative_samples
        self.inner_product_clamp = inner_product_clamp
        self.name = name
        self.algorithm_type = 'SGNS'

        max_weight = 1 / sqrt(embedding_dim)
        torch.nn.init.uniform_(self.embeddings.weight, -max_weight, max_weight)
        torch.nn.init.uniform_(self.out_embeddings.weight, -max_weight, max_weight)

    def forward(self, inputs, positive_outputs, negative_outputs):
        input_embeddings = self.embeddings(inputs)
        positive_output_embeddings = self.out_embeddings(positive_outputs)
        positive_score = torch.clamp(torch.sum(torch.mul(input_embeddings, positive_output_embeddings), dim=1),
                                    min=-self.inner_product_clamp, max=self.inner_product_clamp)
        positive_score_log_sigmoid = - torch.nn.functional.logsigmoid(positive_score)
        negative_output_embeddings = self.out_embeddings(negative_outputs)
        negative_scores = torch.clamp(torch.sum(torch.mul(input_embeddings.unsqueeze(1), negative_output_embeddings),
                                               dim=2), min=-self.inner_product_clamp, max=self.inner_product_clamp)
        negative_scores_log_sigmoid = torch.sum(- torch.nn.functional.logsigmoid(- negative_scores), dim=1)

        return positive_score_log_sigmoid + negative_scores_log_sigmoid


class Elmo(torch.nn.Module):
    def __init__(self, num_layers=2, embedding_dim=data_hyperparameters.WORD_EMBEDDING_DIMENSION, hidden_size=20, vocab_size=data_hyperparameters.VOCAB_SIZE + 2, dropout=0.4, use_packing=True, max_len=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = torch.nn.LSTM(input_size=embedding_dim, dropout=dropout, hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.use_packing = use_packing
        self.max_len = max_len

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len] if self.max_len is not None else inputs
        embeds = self.embedding(input_truncated)
        if self.use_packing:
            input_length = torch.sum(input_truncated != 1, dim=-1)
            embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, input_length, batch_first=True,
                                                                     enforce_sorted=False)
        _, (lstm_hn, _) = self.LSTM(embeds)
        lstm_final_output = torch.flatten(
            lstm_hn.view(self.num_layers, 2, -1, self.hidden_size)[-1, :, :, :].transpose(0, 1),
            start_dim=1)
        out = self.linear(lstm_final_output)
        return torch.nn.functional.log_softmax(out, dim=-1)



def train_w2v(model_name, train_loader, valid_loader, vocab_size=data_hyperparameters.VOCAB_SIZE, distribution=None,
              epochs=data_hyperparameters.WORD_EMBEDDING_EPOCHS,
              embedding_dim=data_hyperparameters.WORD_EMBEDDING_DIMENSION,
              context_size=data_hyperparameters.CONTEXT_SIZE,
              inner_product_clamp=data_hyperparameters.INNER_PRODUCT_CLAMP,
              num_negative_samples=data_hyperparameters.NUM_NEGATIVE_SAMPLES, algorithm='SGNS'):
    train_losses = []
    valid_losses = []
    if algorithm.upper() == 'CBOW':
        model = ContinuousBagOfWords(vocab_size, embedding_dim, context_size, model_name)
        loss_function = torch.nn.NLLLoss()
    else:
        model = SkipGramWithNegativeSampling(vocab_size, embedding_dim, context_size, num_negative_samples,
                                             inner_product_clamp, model_name)
        distribution_tensor = torch.tensor(distribution, dtype=torch.float, device=device)
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)
    write_log('Training on {0} batches and validating on {1} batches'.format(len(train_loader), len(valid_loader)),
              logger)
    for epoch in range(epochs):
        now = datetime.now()
        write_log('Epoch: {0}'.format(epoch), logger)
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            if algorithm.upper() == 'CBOW':
                predictions = model(xb)
                loss = loss_function(predictions, yb)
            else:
                negative_samples = produce_negative_samples(distribution_tensor, num_negative_samples, len(yb))
                loss = torch.mean(model(yb, xb, negative_samples))
            loss.backward()
            total_loss += loss.item()
            optimizer.zero_grad()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        write_log('Training loss: {0}'.format(train_loss), logger)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for xb, yb in valid_loader:
                if algorithm.upper() == 'CBOW':
                    valid_loss += loss_function(model(xb), yb).item()
                elif algorithm.upper() == 'SGNS':
                    negative_samples = produce_negative_samples(distribution_tensor, num_negative_samples, len(yb))
                    loss = model(yb, xb, negative_samples)
                    valid_loss += torch.mean(loss).item()
        valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        write_log('Validation loss: {0}'.format(valid_loss), logger)

        seconds = (datetime.now() - now).total_seconds()
        write_log('Epoch took: {0} seconds'.format(seconds), logger)
        scheduler.step(valid_loss)

    fig, ax = plt.subplots()
    ax.plot(range(epochs), train_losses, label='Training')
    ax.plot(range(epochs), valid_losses, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Learning curve for model {0}'.format(model_name))
    ax.legend()
    plt.savefig('{0}_learning_curve_{1}_{2}_{3}.png'.format(model_name, embedding_dim, algorithm, context_size))

    return model


def save_model_state_w2v(model):
    torch.save(model.state_dict(), model.name + '_' + model.algorithmType + '.pt')
    if model.algorithm_type == 'CBOW':
        model_data = {'embedding_dim': model.embedding_dim, 'context_size': model.context_size}
    else:
        model_data = {'embedding_dim': model.embedding_dim, 'context_size': model.context_size,
                     'num_negative_samples': model.num_negative_samples,
                      'inner_product_clamp': model.inner_product_clamp}
    outfile = open(model.name + '_' + model.algorithmType + '_model_data', 'wb')
    pickle.dump(model_data, outfile)
    outfile.close()
    write_log('Saved model ' + model.name, logger)
    return


def loadModelState_w2v(model_name, algorithm_type,
                       unigram_distribution_power=data_hyperparameters.UNIGRAM_DISTRIBUTION_POWER):
    frequencies = pickle.load(open(FREQS_FILE, "rb"))
    distribution = noise_distribution(frequencies, unigram_distribution_power)
    infile = open(model_name + '_' + algorithm_type + '_model_data', 'rb')
    model_data = pickle.load(infile)
    infile.close()
    if algorithm_type.upper() == 'CBOW':
        model = ContinuousBagOfWords(data_hyperparameters.VOCAB_SIZE, model_data['embeddingDim'],
                                     model_data['contextSize'], model_name)
    else:
        model = SkipGramWithNegativeSampling(data_hyperparameters.VOCAB_SIZE, model_data['embeddingDim'],
                                             model_data['contextSize'], model_data['numNegativeSamples'],
                                             model_data['innerProductClamp'], model_name)
    model.load_state_dict(torch.load(model_name + algorithm_type + '.pt'))
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    write_log('Loaded model {0}'.format(model_name), logger)
    model.eval()
    return frequencies, distribution, model


def train_elmo(model_name, train_loader, valid_loader, vocab_size=data_hyperparameters.VOCAB_SIZE,
               epochs=data_hyperparameters.WORD_EMBEDDING_EPOCHS,
               embedding_dim=data_hyperparameters.WORD_EMBEDDING_DIMENSION):
    pass
