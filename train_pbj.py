import numpy as np
import sklearn
from tqdm import trange
import haiku as hk

import model
from utils import *
import config



def train_early_stopping(data_in, data_out,
                        model_in: model.Model,
                        #val_size=10000,
                        val_size=2,
                        tot_num_steps=1e7,
                        early_stopping_steps=1e6,
                        validation_steps=5e4):
    '''Train the model for a maximum of tot_num_steps gradient steps.
    
    Args:
        data_in: input data, list of jraph graphs
        data_out: output data, list of labels corresponding to inpt data
        model: an initialized model
        val_size: number of validation graphs
        tot_num_steps: maximum number of steps to run, without early stopping
        early_stopping_steps: interval in gradient steps over which the validation loss 
                                is compared for early stopping
        validation_steps: number of steps between evaluations of validation loss
    '''
    train_in, val_in, train_out, val_out = sklearn.model_selection.train_test_split(
        data_in, data_out, train_size=val_size, random_state=0
    )
    reader_train = DataReader(train_in, train_out, model_in.batch_size)
    reader_train.repeat() # training data will be shuffled if all training graphs have been used
    val_loss = [] # will be used as a queue
    for i in trange(int(tot_num_steps), desc=("Training status"), unit="gradient steps"):
        # check early stopping criteria
        if i%validation_steps == 0:
            loss = model_in.test(val_in, val_out)
            print("Validation loss at step {}: {}".format(i, loss))
            val_loss.append(loss)
            # only test for early stopping after the first interval
            if i > early_stopping_steps:
                # stop if new loss higher than loss at beginning of interval
                if loss > val_loss[0]:
                    break
                else:
                    # otherwise delete the element at beginning of queue
                    val_loss.pop(0)

        graphs, labels = next(reader_train)
        graphs = pad_graph_to_nearest_power_of_two(graphs)
        model_in.params, model_in.opt_state, loss = model_in.update(model_in.params, model_in.opt_state, graphs, labels)
        

def main():
    config.N_HIDDEN_C = 64
    print('N_HIDDEN_C: {}'.format(config.N_HIDDEN_C))
    config.AVG_MESSAGE = True
    config.AVG_READOUT = True
    lr = optax.exponential_decay(5*1e-4, 1000, 0.9)
    batch_size = 32
    print('batch size: {}'.format(batch_size))
    model_test = model.Model(lr, batch_size, 5)
    #file_str = 'QM9/graphs_all_labelidx16.csv'
    file_str = 'QM9/graphs_U0K.csv'
    print('Loading data file')
    inputs, outputs, auids = model.get_data_df_csv(file_str)
    inputs = inputs[:10]
    outputs = outputs[:10]
    auids = auids[:10]
    train_in, test_in, train_out, test_out, train_auids, test_auids = sklearn.model_selection.train_test_split(
        inputs, outputs, auids, test_size=0.1, random_state=0
    )
    train_out, mean_train, std_train = model.normalize_targets(train_in, train_out)
    test_out, mean_test, std_test = model.normalize_targets(test_in, test_out)
    outputs, mean_test, std_test = model.normalize_targets(inputs, outputs)
    print('Building model')
    model_test.build(inputs, outputs)
    num_params = hk.data_structures.tree_size(model_test.params)
    byte_size = hk.data_structures.tree_bytes(model_test.params)
    print(f'{num_params} params, size: {byte_size / 1e6:.2f}MB')
    print('Example of labels:')
    print(outputs)
    print('Mean of labels: {}'.format(np.mean(outputs)))
    print('Std of labels: {}'.format(np.std(outputs)))
    
    train_early_stopping(train_in, train_out, model_test)


if __name__ == "__main__":
    main()














