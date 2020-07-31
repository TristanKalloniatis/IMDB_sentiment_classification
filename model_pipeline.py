import torch
import numpy as np
import torchtext
import matplotlib.pyplot as plt
import data_hyperparameters
from datetime import datetime
from log_utils import create_logger, write_log
LOG_FILE = 'model_pipeline'

logger = create_logger(LOG_FILE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # todo: is this needed here?


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


def train(model, train_data, valid_data, epochs=10, loss_function=torch.nn.NLLLoss()):
    # If changing optimiser or scheduler, suggest to record this in model.model_metadata dictionary
    optimiser = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=data_hyperparameters.PATIENCE)
    now_begin_training = datetime.now()
    for epoch in range(epochs):
        now_begin_epoch = datetime.now()
        write_log('Running epoch {0} of {1}'.format(epoch, epochs), logger)
        model.train()
        loss = 0.
        for xb, yb in train_data:
            batch_loss = loss_function(model(xb), yb)
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()
            loss += batch_loss.item() / len(train_data)
        model.train_losses.append(loss)
        write_log('Training loss: {0}'.format(loss), logger)
        model.eval()
        with torch.no_grad():
            loss = sum([loss_function(model(xb), yb).item() for xb, yb in valid_data]) / len(valid_data)
            scheduler.step(loss)
            model.valid_losses.append(loss)
            write_log('Validation loss: {0}'.format(loss), logger)
        model.num_epochs_trained += 1
        write_log('Epoch took {0} seconds'.format((datetime.now() - now_begin_epoch).total_seconds()), logger)
    model.train_time = (datetime.now() - now_begin_training).total_seconds()


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


def report_statistics(model, train_data, valid_data, test_data):
    # todo: report
    model_data = model.get_model_performance_data(train_data, valid_data, test_data)
    pass
