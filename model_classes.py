from abc import ABC
import torch
import data_hyperparameters
import datetime
import matplotlib.pyplot as plt
from math import nan, log
from model_pipeline import prepare_batch_x, prepare_batch_y

def get_accuracy(loader, model):
    with torch.no_grad():
        model.eval()
        accuracy = 0.
        for xb, yb in loader:
            accuracy += model(prepare_batch_x(xb)).argmax(dim=1).eq(prepare_batch_y(yb)).float().mean().item()
    return accuracy / len(loader)


class BaseModelClass(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.num_epochs_trained = 0
        self.train_time = 0.
        self.num_trainable_params = 0
        self.instantiated = datetime.datetime.now()
        self.name = ''
        self.vocab_size = data_hyperparameters.VOCAB_SIZE
        self.tokenizer = data_hyperparameters.TOKENIZER
        self.batch_size = data_hyperparameters.BATCH_SIZE

    def count_parameters(self):
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def cudaify(self):
        if data_hyperparameters.USE_CUDA:
            self.cuda()

    def free(self):
        if data_hyperparameters.USE_CUDA:
            self.cpu()

    def finish_setup(self):
        self.count_parameters()
        self.cudaify()

    def get_model_performance_data(self, train_dataloader, valid_dataloader, test_dataloader):
        final_train_loss = nan if len(self.train_losses) == 0 else self.train_losses[-1]
        final_valid_loss = nan if len(self.valid_losses) == 0 else self.valid_losses[-1]
        train_accuracy = get_accuracy(train_dataloader, self)
        valid_accuracy = get_accuracy(valid_dataloader, self)
        test_accuracy = get_accuracy(test_dataloader, self)
        average_time_per_epoch = nan if self.num_epochs_trained == 0 else self.train_time / self.num_epochs_trained
        model_data = {'name': self.name, 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                      'test_accuracy': test_accuracy, 'total_train_time': self.train_time,
                      'num_epochs': self.num_epochs_trained, 'trainable_params': self.num_trainable_params,
                      'final_train_loss': final_train_loss, 'final_valid_loss': final_valid_loss,
                      'model_created': self.instantiated, 'average_time_per_epoch': average_time_per_epoch,
                      'vocab_size': self.vocab_size, 'tokenizer': self.tokenizer, 'batch_size': self.batch_size}
        return model_data

    def plot_losses(self):
        fig, ax = plt.subplots()
        ax.plot(range(self.num_epochs_trained), self.train_losses, label='Training')
        ax.plot(range(self.num_epochs_trained), self.valid_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning curve for {0}'.format(self.name))
        ax.legend()
        plt.savefig('learning_curve_{0}.png'.format(self.name))


class AverageEmbeddingModel(BaseModelClass, ABC):
    def __init__(self, embedding_dimension=100, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2, name='AVEM'):
        super().__init__()
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.embedding_mean = torch.nn.EmbeddingBag(vocab_size, embedding_dimension)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        embeddings = self.embedding_mean(inputs)
        out = self.linear(embeddings)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LogisticRegressionBOW(BaseModelClass, ABC):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2, name='BOWLR'):
        super().__init__()
        self.name = name
        self.linear = torch.nn.Linear(vocab_size, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        out = self.linear(inputs)
        return torch.nn.functional.log_softmax(out, dim=-1)


class ConvNGram(BaseModelClass, ABC):
    def __init__(self, kernel_size, embedding_dimension=100, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2, name='ConvN'):
        super().__init__()
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.conv = torch.nn.Conv1d(in_channels=embedding_dimension, out_channels=embedding_dimension, kernel_size=kernel_size)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        sequence_length = inputs.shape[1]
        embeds = self.embedding(inputs).view(-1, self.embedding_dimension, sequence_length)
        conv = self.conv(embeds)
        pool = torch.max(conv, dim=-1)[0]
        out = self.linear(pool)
        return torch.nn.functional.log_softmax(out, dim=-1)

class PositionalEncoding(torch.nn.Module):
    # Taken from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        assert d_model % 2 == 0
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # [T, B, d_model] -> [T, B, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(BaseModelClass, ABC):
    def __init__(self, embedding_dimension=100, nhead=4, dim_feedforward=1024, vocab_size=data_hyperparameters.VOCAB_SIZE, pool_type='last', num_categories=2, name='TransformerEncoderLayer'):
        super().__init__()
        assert embedding_dimension % nhead == 0
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.pool_type = pool_type
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.positional_encoder = PositionalEncoding(embedding_dimension)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dimension, nhead=nhead, dim_feedforward=dim_feedforward)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        sequence_length = inputs.shape[1]
        embeds = self.embedding(inputs).view(sequence_length, -1, self.embedding_dimension)
        positional_encodings = self.positional_encoder(embeds)
        transforms = self.encoder_layer(positional_encodings)
        pool = transforms[-1] if self.pool_type == 'last' else torch.max(transforms, dim=0)[0]
        out = self.linear(pool)
        return torch.nn.functional.log_softmax(out, dim=-1)
