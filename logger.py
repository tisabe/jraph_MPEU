import logging
import time
from time import gmtime, strftime

class Logger:
    def __init__(self):
        self.start_time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        logging.basicConfig(filename='logs/' + self.start_time_str + '.log', level=logging.DEBUG)
        logging.info('Starting log')

    def print_start_time(self):
        print(self.start_time_str)

    def log_str(self, str):
        logging.info(str)

    def log_hyperparams(self, data_size, epochs, batch_size, learning_rate):
        logging.info('number of datapoints: {}, epochs: {}, batch size: {}'.format(data_size, epochs, batch_size))
        logging.info('Learning rate: {}'.format(learning_rate))

    def log_loss(self, loss, step):
        logging.info('loss at step {}: {}'.format(step, loss))



