from abc import ABC
import torch
import data_hyperparameters
import datetime
import matplotlib
from math import nan

matplotlib.rcParams['backend'] = 'Qt5Agg'

def get_accuracy(loader, model):
    with torch.no_grad():
        model.eval()
        accuracy = 0.
        for xb, yb in loader:
            accuracy += model(xb).argmax(dim=1).eq(yb).float().mean()
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
        if torch.cuda.is_available():
            self.cuda()

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
        model_data = {'final_train_loss': final_train_loss, 'final_valid_loss': final_valid_loss,
                      'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                      'test_accuracy': test_accuracy, 'name': self.name,
                      'total_train_time': self.train_time, 'num_epochs': self.num_epochs_trained,
                      'trainable_params': self.num_trainable_params, 'model_created': self.instantiated,
                      'average_time_per_epoch': average_time_per_epoch, 'vocab_size': self.vocab_size,
                      'tokenizer': self.tokenizer, 'batch_size': self.batch_size}
        return model_data

    def plot_losses(self): # todo: fix plotting as this does not currently work
        fig, ax = matplotlib.pyplot.subplots()
        ax.plot(range(self.num_epochs_trained), self.train_losses, label='Training')
        ax.plot(range(self.num_epochs_trained), self.valid_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning curve for {0}'.format(self.name))
        ax.legend()
        matplotlib.pyplot.savefig('learning_curve_{0}.png'.format(self.name))


class AverageEmbeddingModel(BaseModelClass, ABC):
    def __init__(self, embedding_dimension=100, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.name = 'AVEM'
        self.embedding_dimension = embedding_dimension
        self.embedding_mean = torch.nn.EmbeddingBag(vocab_size, embedding_dimension)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        embeddings = self.embedding_mean(inputs)
        out = self.linear(embeddings)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LogisticRegressionBOW(BaseModelClass, ABC):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.name = 'BOWLR'
        self.linear = torch.nn.Linear(vocab_size, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        out = self.linear(inputs)
        return torch.nn.functional.log_softmax(out, dim=-1)
