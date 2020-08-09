import torch
import data_hyperparameters
from datetime import datetime
import os
import csv
from log_utils import create_logger, write_log

LOG_FILE = 'model_pipeline'
logger = create_logger(LOG_FILE)


def train(model, train_data, valid_data, epochs=10):
    loss_function = torch.nn.NLLLoss()
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    optimiser = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=data_hyperparameters.PATIENCE)
    now_begin_training = datetime.now()
    start_epoch = model.num_epochs_trained
    for epoch in range(start_epoch, epochs + start_epoch):
        now_begin_epoch = datetime.now()
        write_log('Running epoch {0} of {1}'.format(epoch + 1, epochs + start_epoch), logger)
        model.train()
        loss = 0.
        for bidx, (xb, yb) in enumerate(train_data):
            print(bidx)
            if data_hyperparameters.USE_CUDA:
                yb = yb.cuda()
                xb = (x.cuda() for x in xb) if isinstance(xb, tuple) else xb.cuda()
            outputs = model(xb)
            batch_loss = loss_function(outputs, yb)
            loss += batch_loss.item() / len(train_data)
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()
        model.train_losses.append(loss)
        write_log('Training loss: {0}'.format(loss), logger)
        model.eval()
        with torch.no_grad():
            loss = 0.
            for xb, yb in valid_data:
                if data_hyperparameters.USE_CUDA:
                    yb = yb.cuda()
                    xb = (x.cuda() for x in xb) if isinstance(xb, tuple) else xb.cuda()
                outputs = model(xb)
                loss += loss_function(outputs, yb).item() / len(valid_data)
        model.valid_losses.append(loss)
        scheduler.step(loss)
        write_log('Validation loss: {0}'.format(loss), logger)
        model.num_epochs_trained += 1
        write_log('Epoch took {0} seconds'.format((datetime.now() - now_begin_epoch).total_seconds()), logger)
    model.train_time += (datetime.now() - now_begin_training).total_seconds()
    if data_hyperparameters.USE_CUDA:
        model.cpu()


def report_statistics(model, train_data, valid_data, test_data):
    model_data = model.get_model_performance_data(train_data, valid_data, test_data)
    if not os.path.isfile(data_hyperparameters.STATISTICS_FILE):
        with open(data_hyperparameters.STATISTICS_FILE, 'w') as f:
            w = csv.DictWriter(f, model_data.keys())
            w.writeheader()
            w.writerow(model_data)
    else:
        with open(data_hyperparameters.STATISTICS_FILE, 'a') as f:
            w = csv.DictWriter(f, model_data.keys())
            w.writerow(model_data)
