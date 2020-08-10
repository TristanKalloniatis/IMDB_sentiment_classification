# IMDB_sentiment_classification
Compare models for sentiment classification using PyTorch

Usage: Add a class definition to model_classes.py

Either modify and run run_experiment.py, or in a notebook:

import model_pipeline, data_downloader
train_data, valid_data, test_data = data_downloader.get_dataloaders() # assuming model uses default dataloaders

model_pipeline.train(model, train_data, valid_data) # can add an epochs parameter. Can rerun this cell and training will continue from the previous end point

model.plot_losses()

model_classes.get_accuracy(train_data, model)

model_classes.get_accuracy(valid_data, model)

model_pipeline.report_statistics(model, train_data, valid_data, test_data) # when satisfied with the state of the model
