import torch
import data_hyperparameters
from datetime import datetime
import os
import csv
from log_utils import create_logger, write_log
from model_classes import get_accuracy
from pickle import dump, load

LOG_FILE = 'model_pipeline'
logger = create_logger(LOG_FILE)


def train(model, train_data, valid_data, epochs=data_hyperparameters.EPOCHS, patience=data_hyperparameters.PATIENCE, report_accuracy_every=None):
    loss_function = torch.nn.NLLLoss()
    if data_hyperparameters.USE_CUDA:
        model.cuda()
    optimiser = torch.optim.Adam(model.parameters()) if model.latest_scheduled_lr is None else torch.optim.Adam(model.parameters(), lr=model.latest_scheduled_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=patience)
    now_begin_training = datetime.now()
    start_epoch = model.num_epochs_trained
    for epoch in range(start_epoch, epochs + start_epoch):
        now_begin_epoch = datetime.now()
        write_log('Running epoch {0} of {1}'.format(epoch + 1, epochs + start_epoch), logger)
        model.train()
        model.latest_scheduled_lr = optimiser.param_groups[0]['lr']
        loss = 0.
        for xb, yb in train_data:
            batch_loss = loss_function(model(xb), yb)
            loss += batch_loss.item() / len(train_data)
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()
        model.train_losses.append(loss)
        write_log('Training loss: {0}'.format(loss), logger)
        if report_accuracy_every is not None:
            if epoch + 1 % report_accuracy_every == 0:
                accuracy = get_accuracy(train_data, model)
                write_log('Training accuracy: {0}'.format(accuracy), logger)
                model.train_accuracies[epoch + 1] = accuracy
        model.eval()
        with torch.no_grad():
            loss = sum([loss_function(model(xb), yb).item() for xb, yb in valid_data]) / len(valid_data)
        model.valid_losses.append(loss)
        scheduler.step(loss)
        write_log('Validation loss: {0}'.format(loss), logger)
        if report_accuracy_every is not None:
            if epoch + 1 % report_accuracy_every == 0:
                accuracy = get_accuracy(valid_data, model)
                write_log('Validation accuracy: {0}'.format(accuracy), logger)
                model.valid_accuracies[epoch + 1] = accuracy
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

def save_model(model):
    torch.save(model.state_dict(), model.name + '.pt')
    model_data = {'train_losses': model.train_losses, 'valid_losses': model.valid_losses,
                  'num_epochs_trained': model.num_epochs_trained, 'latest_scheduled_lr': model.latest_scheduled_lr,
                  'train_time': model.train_time, 'num_trainable_params': model.num_trainable_params,
                  'instantiated': model.instantiated, 'name': model.name, 'vocab_size': model.vocab_size,
                  'tokenizer': model.tokenizer, 'batch_size': model.batch_size,
                  'train_accuracies': model.train_accuracies, 'valid_accuracies': model.valid_accuracies}
    outfile = open(model.name + '_model_data', 'wb')
    dump(model_data, outfile)
    outfile.close()

def load_model_state(model, model_name):
    model.load_state_dict(torch.load(model_name + '.pt'))
    write_log('Loaded model {0} weights'.format(model_name), logger)
    infile = open(model_name + '_model_data', 'rb')
    model_data = load(infile)
    infile.close()
    model.train_losses = model_data['train_losses']
    model.valid_losses = model_data['valid_losses']
    model.num_epochs_trained = model_data['num_epochs_trained']
    model.latest_scheduled_lr = model_data['latest_scheduled_lr']
    model.train_time = model_data['train_time']
    model.num_trainable_params = model_data['num_trainable_params']
    model.instantiated = model_data['instantiated']
    model.name = model_data['name']
    model.vocab_size = model_data['vocab_size']
    model.tokenizer = model_data['tokenizer']
    model.batch_size = model_data['batch_size']
    model.train_accuracies = model_data['train_accuracies']
    model.valid_accuracies = model_data['valid_accuracies']
    write_log('Loaded model {0} state'.format(model_name), logger)
