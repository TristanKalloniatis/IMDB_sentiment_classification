import model_classes
import model_pipeline
import data_hyperparameters
import data_downloader
from log_utils import create_logger, write_log
LOG_FILE = 'run_experiment'
logger = create_logger(LOG_FILE)

train_data, valid_data, test_data = data_downloader.get_dataloaders()
# Note: this will not work directly with LogisticRegressionBOW as this trains with a different dataset
#Also should do naive Bayes separately (with naive_bayes.report_statistics() since it's based on a different paradigm
#Usage: add models to the list below
models = [model_classes.TransformerEncoderLayer(max_len=200, name='TransformerEncoderLayer_200')]
for model in models:
    write_log('Running experiment for {0}'.format(model.name), logger)
    model_pipeline.train(model=model, train_data=train_data, valid_data=valid_data)
    model.plot_losses()
    model_pipeline.report_statistics(model=model, train_data=train_data, valid_data=valid_data, test_data=test_data)
