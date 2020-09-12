from abc import ABC
import torch
import data_hyperparameters
import datetime
import matplotlib.pyplot as plt
from math import nan, log
import os

if not os.path.exists('learning_curves/'):
    os.mkdir('learning_curves/')


def get_accuracy(loader, model, also_report_model_confidences=False):
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    model.eval()
    with torch.no_grad():
        accuracy = 0.
        correct_prediction_probs = 0.
        incorrect_prediction_probs = 0.
        num_correct = 0
        num_incorrect = 0
        for xb, yb in loader:
            if data_hyperparameters.USE_CUDA and not data_hyperparameters.STORE_DATA_ON_GPU_IF_AVAILABLE:
                xb = xb.cuda()
                yb = yb.cuda()
            model_output = model(xb)
            accuracy += model_output.argmax(dim=1).eq(yb).float().mean().item()
            if also_report_model_confidences:
                log_probs, predictions = torch.max(model_output, dim=-1)
                probs = torch.exp(log_probs)
                correct_predictions_mask = torch.where(predictions == yb, torch.ones_like(yb), torch.zeros_like(yb))
                num_correct += torch.sum(correct_predictions_mask).item()
                num_incorrect += torch.sum(1 - correct_predictions_mask).item()
                correct_prediction_probs += torch.sum(correct_predictions_mask * probs).item()
                incorrect_prediction_probs += torch.sum((1 - correct_predictions_mask) * probs).item()
    if also_report_model_confidences:
        return accuracy / len(loader), correct_prediction_probs / num_correct, incorrect_prediction_probs / num_incorrect
    else:
        return accuracy / len(loader)


class BaseModelClass(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []
        self.num_epochs_trained = 0
        self.latest_scheduled_lr = None
        self.lr_history = []
        self.train_time = 0.
        self.num_trainable_params = 0
        self.instantiated = datetime.datetime.now()
        self.name = ''
        self.vocab_size = data_hyperparameters.VOCAB_SIZE
        self.tokenizer = data_hyperparameters.TOKENIZER
        self.batch_size = data_hyperparameters.BATCH_SIZE
        self.train_accuracies = {}
        self.valid_accuracies = {}
        self.train_correct_confidences = {}
        self.train_incorrect_confidences = {}
        self.valid_correct_confidences = {}
        self.valid_incorrect_confidences = {}

    def count_parameters(self):
        self.num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def finish_setup(self):
        self.count_parameters()

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

    def plot_losses(self, include_lrs=True):
        fig, ax = plt.subplots()
        ax.plot(range(self.num_epochs_trained), self.train_losses, label='Training')
        ax.plot(range(self.num_epochs_trained), self.valid_losses, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning curve for {0}'.format(self.name))
        ax.legend()
        plt.savefig('learning_curves/learning_curve_{0}.png'.format(self.name))

        if len(self.train_accuracies) != 0:
            fig, ax = plt.subplots()
            epochs = list(self.train_accuracies.keys())
            train_accuracies = list(self.train_accuracies.values())
            valid_accuracies = list(self.valid_accuracies.values())
            ax.scatter(epochs, train_accuracies, label='Training')
            ax.scatter(epochs, valid_accuracies, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracies for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/accuracies_{0}.png'.format(self.name))

        if len(self.train_correct_confidences) != 0:
            fig, ax = plt.subplots()
            epochs = list(self.train_correct_confidences.keys())
            train_correct_confidences = list(self.train_correct_confidences.values())
            train_incorrect_confidences = list(self.train_incorrect_confidences.values())
            valid_correct_confidences = list(self.valid_correct_confidences.values())
            valid_incorrect_confidences = list(self.valid_incorrect_confidences.values())
            ax.scatter(epochs, train_correct_confidences, label='Training (correct predictions)')
            ax.scatter(epochs, valid_correct_confidences, label='Validation (correct predictions)')
            ax.scatter(epochs, train_incorrect_confidences, label='Training (incorrect predictions)')
            ax.scatter(epochs, valid_incorrect_confidences, label='Validation (incorrect predictions)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Confidence')
            ax.set_title('Confidences for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/confidences_{0}.png'.format(self.name))

        if include_lrs:
            fig, ax = plt.subplots()
            ax.plot(range(self.num_epochs_trained), self.lr_history, label='Learning rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning rate')
            ax.set_title('Learning rates for {0}'.format(self.name))
            ax.legend()
            plt.savefig('learning_curves/learning_rates_{0}.png'.format(self.name))


class AverageEmbeddingModel(BaseModelClass, ABC):
    def __init__(self, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, num_categories=2, name='AVEM'):
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


# WARNING: THIS CLASS EXPECTS A DIFFERENT DATA FORMAT SO SHOULD NOT BE USED WITH DEFAULT DATALOADER
class BOWLogisticRegression(BaseModelClass, ABC):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2, name='BOWLR'):
        super().__init__()
        self.name = name
        self.linear = torch.nn.Linear(vocab_size, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        out = self.linear(inputs)
        return torch.nn.functional.log_softmax(out, dim=-1)


class ConvNGram(BaseModelClass, ABC):
    def __init__(self, kernel_size, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 dropout=data_hyperparameters.DROPOUT, embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, num_features=data_hyperparameters.NUM_CONV_FEATURES,
                 num_categories=2, name='ConvN'):
        super().__init__()
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.conv = torch.nn.Conv1d(in_channels=embedding_dimension, out_channels=num_features,
                                    kernel_size=kernel_size)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        embeds = self.embedding(inputs).transpose(1, 2)
        conv = self.conv(self.embedding_dropout(embeds))
        pool = torch.max(conv, dim=-1)[0]
        out = self.linear(self.dropout_softmax(pool))
        return torch.nn.functional.log_softmax(out, dim=-1)


class ConvMultiGram(BaseModelClass, ABC):
    def __init__(self, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 dropout=data_hyperparameters.DROPOUT, embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, num_features=data_hyperparameters.NUM_CONV_FEATURES,
                 num_categories=2, name='ConvMultiGram'):
        super().__init__()
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.conv2 = torch.nn.Conv1d(in_channels=embedding_dimension, out_channels=num_features, kernel_size=2)
        self.conv3 = torch.nn.Conv1d(in_channels=embedding_dimension, out_channels=num_features, kernel_size=3)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(embedding_dimension + 2 * num_features, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        embeds = self.embedding(inputs).transpose(1, 2)
        dropped_embeds = self.embedding_dropout(embeds)
        conv2 = self.conv2(dropped_embeds)
        conv3 = self.conv3(dropped_embeds)
        pool1 = torch.max(dropped_embeds, dim=-1)[0]
        pool2 = torch.max(conv2, dim=-1)[0]
        pool3 = torch.max(conv3, dim=-1)[0]
        pool_cat = torch.cat((pool1, pool2, pool3), dim=-1)
        pool_dropout = self.dropout_softmax(pool_cat)
        out = self.linear(pool_dropout)
        return torch.nn.functional.log_softmax(out, dim=-1)


class GRUClassifier(BaseModelClass, ABC):
    def __init__(self, num_layers, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT, dropout=data_hyperparameters.DROPOUT,
                 recurrent_dropout=data_hyperparameters.RECURRENT_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, max_len=None,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, bidirectional=False, num_categories=2, use_packing=False,
                 batch_first=True, name='GRU'):
        super().__init__()
        self.bidirectional = bidirectional
        self.max_len = max_len
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size_scaled = hidden_size * self.num_directions
        self.use_packing = use_packing
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(input_size=embedding_dimension, hidden_size=hidden_size, num_layers=num_layers,
                                bidirectional=bidirectional, batch_first=batch_first, dropout=recurrent_dropout)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(self.hidden_size_scaled, num_categories)
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len] if self.max_len is not None else inputs
        embeds = self.embedding(input_truncated)
        dropped_embeds = self.embedding_dropout(embeds)
        if not self.batch_first:
            dropped_embeds = dropped_embeds.transpose(0, 1)
        if self.use_packing:
            input_length = torch.sum(input_truncated != 1, dim=-1)
            dropped_embeds = torch.nn.utils.rnn.pack_padded_sequence(dropped_embeds, input_length, enforce_sorted=False,
                                                                     batch_first=self.batch_first)
        _, gru_hn = self.GRU(dropped_embeds)
        gru_final_output = torch.flatten(
            gru_hn.view(self.num_layers, self.num_directions, -1, self.hidden_size)[-1, :, :, :].transpose(0, 1),
            start_dim=1)
        gru_final_dropout = self.dropout_softmax(gru_final_output)
        out = self.linear(gru_final_dropout)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LSTMClassifier(BaseModelClass, ABC):
    def __init__(self, num_layers, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 dropout=data_hyperparameters.DROPOUT, embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT,
                 recurrent_dropout=data_hyperparameters.RECURRENT_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, max_len=None,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, bidirectional=False, num_categories=2, use_packing=False,
                 batch_first=True, name='LSTM'):
        super().__init__()
        self.bidirectional = bidirectional
        self.max_len = max_len
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size_scaled = hidden_size * self.num_directions
        self.use_packing = use_packing
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=bidirectional, batch_first=batch_first, dropout=recurrent_dropout)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(self.hidden_size_scaled, num_categories)
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len] if self.max_len is not None else inputs
        embeds = self.embedding(input_truncated)
        dropped_embeds = self.embedding_dropout(embeds)
        if not self.batch_first:
            dropped_embeds = dropped_embeds.transpose(0, 1)
        if self.use_packing:
            input_length = torch.sum(input_truncated != 1, dim=-1)
            dropped_embeds = torch.nn.utils.rnn.pack_padded_sequence(dropped_embeds, input_length, enforce_sorted=False,
                                                                     batch_first=self.batch_first)
        _, (lstm_hn, _) = self.LSTM(dropped_embeds)
        lstm_final_output = torch.flatten(
            lstm_hn.view(self.num_layers, self.num_directions, -1, self.hidden_size)[-1, :, :, :].transpose(0, 1),
            start_dim=1)
        lstm_final_dropout = self.dropout_softmax(lstm_final_output)
        out = self.linear(lstm_final_dropout)
        return torch.nn.functional.log_softmax(out, dim=-1)


class GRUClassifierWithDotProductAttention(BaseModelClass, ABC):
    def __init__(self, num_layers, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT, dropout=data_hyperparameters.DROPOUT,
                 recurrent_dropout=data_hyperparameters.RECURRENT_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, max_len=None,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, num_categories=2, name='GRUWithDotProductAttention'):
        super().__init__()
        self.max_len = max_len
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.GRU = torch.nn.GRU(input_size=embedding_dimension, hidden_size=hidden_size, num_layers=num_layers,
                                batch_first=True, dropout=recurrent_dropout)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(self.hidden_size, num_categories)
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len] if self.max_len is not None else inputs
        embeds = self.embedding(input_truncated)
        dropped_embeds = self.embedding_dropout(embeds)
        gru_output, gru_hn = self.GRU(dropped_embeds)
        similarity_scores = torch.nn.functional.softmax(torch.bmm(gru_output, gru_hn[-1, :, :].unsqueeze(-1)).squeeze(),
                                                        dim=-1)
        gru_final_output = torch.sum(similarity_scores.unsqueeze(-1) * gru_output, dim=1)
        gru_final_dropout = self.dropout_softmax(gru_final_output)
        out = self.linear(gru_final_dropout)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LSTMClassifierWithDotProductAttention(BaseModelClass, ABC):
    def __init__(self, num_layers, embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 embedding_dropout=data_hyperparameters.EMBEDDING_DROPOUT, dropout=data_hyperparameters.DROPOUT,
                 recurrent_dropout=data_hyperparameters.RECURRENT_DROPOUT,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, max_len=None,
                 hidden_size=data_hyperparameters.HIDDEN_SIZE, num_categories=2, name='LSTMWithDotProductAttention'):
        super().__init__()
        self.max_len = max_len
        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        self.LSTM = torch.nn.LSTM(input_size=embedding_dimension, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=recurrent_dropout)
        self.dropout_softmax = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(self.hidden_size, num_categories)
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len] if self.max_len is not None else inputs
        embeds = self.embedding(input_truncated)
        dropped_embeds = self.embedding_dropout(embeds)
        lstm_output, (lstm_hn, _) = self.LSTM(dropped_embeds)
        similarity_scores = torch.nn.functional.softmax(torch.bmm(lstm_output, lstm_hn[-1, :, :].unsqueeze(-1)).squeeze(),
                                                        dim=-1)
        lstm_final_output = torch.sum(similarity_scores.unsqueeze(-1) * lstm_output, dim=1)
        lstm_final_dropout = self.dropout_softmax(lstm_final_output)
        out = self.linear(lstm_final_dropout)
        return torch.nn.functional.log_softmax(out, dim=-1)

class PositionalEncoding(torch.nn.Module):
    # Modified from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    def __init__(self, d_model, dropout=data_hyperparameters.POSITIONAL_ENCODING_DROPOUT,
                 max_len=data_hyperparameters.TRANSFORMER_MAX_LEN):
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
    def __init__(self, max_len=data_hyperparameters.TRANSFORMER_MAX_LEN,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 nhead=data_hyperparameters.TRANSFORMER_NHEAD,
                 dim_feedforward=data_hyperparameters.TRANSFORMER_DIM_FEEDFORWARD,
                 dropout=data_hyperparameters.TRANSFORMER_DROPOUT,
                 positional_encoding_dropout=data_hyperparameters.POSITIONAL_ENCODING_DROPOUT, mask_pad_tokens=True,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, pool_type='mean', num_categories=2,
                 name='TransformerEncoderLayer'):
        super().__init__()
        assert embedding_dimension % nhead == 0
        self.max_len = max_len
        self.mask_pad_tokens = mask_pad_tokens
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.pool_type = pool_type
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.positional_encoder = PositionalEncoding(embedding_dimension, max_len=max_len,
                                                     dropout=positional_encoding_dropout)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dimension, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len]
        embeds = self.embedding(input_truncated).transpose(0, 1)
        positional_encodings = self.positional_encoder(embeds)
        if self.mask_pad_tokens:
            src_key_padding_mask = (input_truncated == 1)
            transforms = self.encoder_layer(positional_encodings, src_key_padding_mask=src_key_padding_mask)
            if self.pool_type == 'mean':
                non_pad = (1 - src_key_padding_mask.float()).transpose(0, 1).unsqueeze(-1)
                non_pad_sum = torch.sum(non_pad, dim=0)
                non_pad_transforms_sum = torch.sum(non_pad * transforms, dim=0)
                pool = non_pad_transforms_sum / non_pad_sum
            else:
                masked_transforms = transforms.masked_fill(src_key_padding_mask.transpose(0,1).unsqueeze(-1), -float('inf'))
                pool = torch.max(masked_transforms, dim=0)[0]
        else:
            transforms = self.encoder_layer(positional_encodings)
            pool = torch.mean(transforms, dim=0) if self.pool_type == 'mean' else torch.max(transforms, dim=0)[0]
        out = self.linear(pool)
        return torch.nn.functional.log_softmax(out, dim=-1)


class TransformerEncoder(BaseModelClass, ABC):
    def __init__(self, num_layers, max_len=data_hyperparameters.TRANSFORMER_MAX_LEN,
                 embedding_dimension=data_hyperparameters.EMBEDDING_DIMENSION,
                 nhead=data_hyperparameters.TRANSFORMER_NHEAD,
                 dim_feedforward=data_hyperparameters.TRANSFORMER_DIM_FEEDFORWARD,
                 dropout=data_hyperparameters.TRANSFORMER_DROPOUT,
                 positional_encoding_dropout=data_hyperparameters.POSITIONAL_ENCODING_DROPOUT, mask_pad_tokens=True,
                 vocab_size=data_hyperparameters.VOCAB_SIZE + 2, pool_type='mean', num_categories=2,
                 name='TransformerEncoder'):
        super().__init__()
        assert embedding_dimension % nhead == 0
        self.max_len = max_len
        self.mask_pad_tokens = mask_pad_tokens
        self.name = name
        self.embedding_dimension = embedding_dimension
        self.pool_type = pool_type
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dimension)
        self.positional_encoder = PositionalEncoding(embedding_dimension, max_len=max_len,
                                                     dropout=positional_encoding_dropout)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dimension, nhead=nhead,
                                                              dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)
        self.finish_setup()

    def forward(self, inputs):
        input_truncated = inputs[:, :self.max_len]
        embeds = self.embedding(input_truncated).transpose(0, 1)
        positional_encodings = self.positional_encoder(embeds)
        if self.mask_pad_tokens:
            src_key_padding_mask = (input_truncated == 1)
            transforms = self.encoder(positional_encodings, src_key_padding_mask=src_key_padding_mask)
            if self.pool_type == 'mean':
                non_pad = (1 - src_key_padding_mask.float()).transpose(0, 1).unsqueeze(-1)
                non_pad_sum = torch.sum(non_pad, dim=0)
                non_pad_transforms_sum = torch.sum(non_pad * transforms, dim=0)
                pool = non_pad_transforms_sum / non_pad_sum
            else:
                masked_transforms = transforms.masked_fill(src_key_padding_mask.transpose(0,1).unsqueeze(-1), -float('inf'))
                pool = torch.max(masked_transforms, dim=0)[0]
        else:
            transforms = self.encoder(positional_encodings)
            pool = torch.mean(transforms, dim=0) if self.pool_type == 'mean' else torch.max(transforms, dim=0)[0]
        out = self.linear(pool)
        return torch.nn.functional.log_softmax(out, dim=-1)


class MyTemplateModelClass(BaseModelClass, ABC):
    def __init__(self, model_parameters, name='template', num_categories=2):
        super().__init__()
        self.model_parameters = model_parameters
        self.name = name
        self.finish_setup()

    def forward(self, inputs):
        pass
