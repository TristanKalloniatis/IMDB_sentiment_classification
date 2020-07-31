from datetime import datetime
import logging
from sys import stdout


def write_log(message, log_object):
    timestamp = datetime.now()
    log_object.info('[{0}]: {1}'.format(timestamp, message))
    return


def create_logger(log_filename):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, stream=stdout)
    logger.addHandler(logging.FileHandler('log_' + log_filename + '.txt'))
    return logger
