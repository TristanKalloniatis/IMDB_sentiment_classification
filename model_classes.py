import torch
import data_hyperparameters

class AverageEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dimension=100, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.embedding_mean = torch.nn.EmbeddingBag(vocab_size, embedding_dimension)
        self.linear = torch.nn.Linear(embedding_dimension, num_categories)

    def forward(self, inputs):
        embeddings = self.embedding_mean(inputs)
        out = self.linear(embeddings)
        return torch.nn.functional.log_softmax(out, dim=-1)


class LogisticRegressionBOW(torch.nn.Module):
    def __init__(self, vocab_size=data_hyperparameters.VOCAB_SIZE, num_categories=2):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size, num_categories)

    def forward(self, inputs):
        out = self.linear(inputs)
        return torch.nn.functional.log_softmax(out, dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)