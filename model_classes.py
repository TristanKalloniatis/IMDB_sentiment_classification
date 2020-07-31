from abc import ABC
import torch
import data_hyperparameters
import datetime
import matplotlib.pyplot as plt


class BaseModelClass(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.num_epochs_trained = 0
        self.train_time = 0.
        self.num_trainable_params = 0
        self.instantiated = datetime.datetime.now()
        self.model_metadata = {}
        self.name = ''
        self.vocab_size = data_hyperparameters.VOCAB_SIZE
        self.tokenizer = data_hyperparameters.TOKENIZER

    def count_parameters(self):
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_performance_data(self, train_dataloader, valid_dataloader, test_dataloader):
        final_train_loss = self.train_losses[-1]
        final_valid_loss = self.valid_losses[-1]
        train_accuracy = 0  # todo: calculate these
        valid_accuracy = 0
        test_accuracy = 0
        average_time_per_epoch = self.train_time / self.num_epochs_trained
        model_data = {'final_train_loss': final_train_loss, 'final_valid_loss': final_valid_loss,
                      'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
                      'test_accuracy': test_accuracy, 'name': self.name,
                      'total_train_time': self.train_time, 'num_epochs': self.num_epochs_trained,
                      'trainable_params': self.num_trainable_params, 'model_created': self.instantiated,
                      'average_time_per_epoch': average_time_per_epoch, 'vocab_size': self.vocab_size,
                      'tokenizer': self.tokenizer}
        for key in self.model_metadata:
            model_data[key] = self.model_metadata[key]
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
    def __init__(self, embedding_dimension=100, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.count_parameters()
        self.name = 'AVEM'
        self.embedding_dimension = embedding_dimension
        self.embedding_mean = torch.nn.EmbeddingBag(vocab_size, embedding_dimension)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.count_parameters()

    def forward(self, inputs):
        embeddings = self.embedding_mean(inputs)
        out = self.linear(embeddings)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LogisticRegressionBOW(BaseModelClass, ABC):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.name = 'BOWLR'
        self.linear = torch.nn.Linear(vocab_size, num_categories)
        self.count_parameters()

    def forward(self, inputs):
        out = self.linear(inputs)
        return torch.nn.functional.log_softmax(out, dim=-1)
