import torch
import numpy as np
import torchtext
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime


def compute_accuracy(preds, actuals):
    num_correct = 0
    for i in range(len(preds)):
        if preds[i] == actuals[i]:
            num_correct += 1
    return num_correct / len(preds)


def dataloader_predict(loader, model):
    preds = []
    with torch.no_grad():
        model.eval()
        for xb, _ in loader:
            preds = preds + torch.argmax(model(xb), dim=1).tolist()
    return preds


def train(model, train_data, valid_data, epochs=100, loss_function=torch.nn.NLLLoss()):
    optimiser = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5)
    train_losses = []
    valid_losses = []
    for epoch in tqdm.notebook.tnrange(epochs): # todo: change this
        model.train()
        loss = 0
        for xb, yb in train_data:
            batch_loss = loss_function(model(xb), yb)
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()
            loss += batch_loss.item() / len(train_data)
        train_losses.append(loss)
        model.eval()
        with torch.no_grad():
            loss = sum([loss_function(model(xb), yb).item() for xb, yb in valid_data]) / len(valid_data)
            scheduler.step(loss)
            valid_losses.append(loss)
    return train_losses, valid_losses


def evaluate_dataloder_model(model, train_data, valid_data, test_data):
    train_preds = dataloader_predict(train_data, model)
    valid_preds = dataloader_predict(valid_data, model)
    test_preds = dataloader_predict(test_data, model)
    train_actuals = []
    valid_actuals = []
    test_actuals = []
    for _, yb in train_data:
        train_actuals += yb.tolist()
    for _, yb in valid_data:
        valid_actuals += yb.tolist()
    for _, yb in test_data:
        test_actuals += yb.tolist()
    print('Train accuracy: {0}'.format(compute_accuracy(train_preds, train_actuals)))
    print('Valid accuracy: {0}'.format(compute_accuracy(valid_preds, valid_actuals)))
    print('Test accuracy: {0}'.format(compute_accuracy(test_preds, test_actuals)))


def report_statistics():
    # todo: report time run, time per epoch
    pass
