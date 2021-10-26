import jax
import jax.numpy as jnp
from numpy.lib import type_check
import jraph
import numpy as np
import haiku as hk
import functools
import optax
import pandas
from tqdm import trange
import sklearn

# import custom functions
from graph_net_fn import *
from utils import *
import config


def get_data_df_csv(file_str, include_no_edge_graphs=False):
    '''Import data from a pandas.DataFrame saved as csv. 
    Return as inputs, outputs (e.g. train_inputs, train_outputs)'''
    df = pandas.read_csv(file_str)
    inputs = []
    outputs = []
    for index, row in df.iterrows():
        #print(index)
        nodes = str_to_array_replace(row['nodes'])
        #nodes = np.reshape(nodes, (-1,1)).astype(np.float32)
        #print(nodes)
        #print(type(nodes))
        #print(row['senders'])
        senders = str_to_array_replace(row['senders'])
        #senders = row['senders']
        receivers = str_to_array_replace(row['receivers'])
        #receivers = row['receivers']
        #print(index)
        #print(row['edges'])
        edges = str_to_array_float(row['edges'])

        if (not len(edges)==0) or include_no_edge_graphs:
            graph = jraph.GraphsTuple(
                n_node=np.asarray([len(nodes)]),
                n_edge=np.asarray([len(senders)]),
                nodes=nodes, edges=edges,
                globals=None,
                senders=np.asarray(senders), receivers=np.asarray(receivers))
            inputs.append(graph)
            outputs.append(row['label'])

    return inputs, outputs

def make_result_csv(x, y, path):
    '''Print predictions x versus labels y in a csv at path.'''
    dict_res = {'x': np.array(x).flatten(), 'y': np.array(y).flatten()}
    df = pandas.DataFrame(data=dict_res)
    df.to_csv(path)

def get_highest_atomic_number(input_graphs):
    '''Return the highest atomic number in the node features of any graph in input_graphs.'''
    max_num = 0
    for i in range(len(input_graphs)):
        graph = input_graphs[i]
        nodes = graph.nodes
        max_local = max(nodes)
        if max_local > max_num:
            max_num = max_local
    return int(max_num)


class Model:
    '''Make a MPEU model.'''
    def __init__(self, learning_rate, batch_size, epochs):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.built = False # set to false until model has been built
        self.set_train_logging()
        self.data_in = None
        self.data_out = None
        self.train_loss_arr = []
        self.test_loss_arr = []

    def compute_loss(self, params, graph, label, net):
        """Computes loss."""
        pred_graph = net.apply(params, graph)
        preds = pred_graph.globals

        # one graph was added to pad nodes and edges, so globals will also be padded by one
        # masking is not needed so long as the padded graph also has a zero global array after update
        label_padded = jnp.pad(label, ((0, 1), (0, 0)))

        loss = jnp.sum(jnp.abs(preds - label_padded))
        return loss

    def build(self, inputs, outputs):
        '''Initialize optimiser and model parameters'''
        graph_example = inputs[0]
        print(graph_example)
        label_example = outputs[0]
        print(label_example)
        if type(label_example) is float:
            config.LABEL_SIZE = 1
        else:
            config.LABEL_SIZE = label_example.shape()

        config.MAX_ATOMIC_NUMBER = get_highest_atomic_number(inputs)
        print(config.MAX_ATOMIC_NUMBER)

        self.net = hk.without_apply_rng(hk.transform(net_fn)) # initializing haiku MLP layers
        self.params = self.net.init(jax.random.PRNGKey(42), graph_example)
        opt_init, self.opt_update = optax.adam(self.learning_rate)
        self.opt_state = opt_init(self.params)

        self.compute_loss_fn = functools.partial(self.compute_loss, net=self.net)
        self.compute_loss_fn = jax.jit(jax.value_and_grad(
                                    self.compute_loss_fn))

        self.built = True
        


    #@jax.jit
    def update(self,
            params: hk.Params,
            opt_state: optax.OptState,
            graph: jraph.GraphsTuple,
            label: jnp.ndarray,
        ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        loss, grad = self.compute_loss_fn(params, graph, label)
        updates, self.opt_state = self.opt_update(grad, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss

    def train_epoch(self, inputs, outputs, idx_epoch):
        '''Train the model for a single epoch.'''
        total_num_graphs = len(inputs)
        num_training_steps_per_epoch = total_num_graphs // self.batch_size

        loss_sum = 0
        check_sum = 0 # check if all graphs have been used in epoch
        for i in trange(num_training_steps_per_epoch, desc=("epoch " + str(idx_epoch)), unit="gradient steps"):
            graphs = []
            labels = []
            for idx_batch in range(self.batch_size):
                # loop over the batch size to create a jraph batch from inputs and outputs
                graph = inputs[i*self.batch_size+idx_batch]
                label = outputs[i*self.batch_size+idx_batch]
                graphs.append(graph)
                labels.append([label])
                check_sum += 1
            # return jraph.batch(graphs), np.concatenate(labels, axis=0)
            #graphs = inputs[i*self.batch_size:i*self.batch_size+self.batch_size]
            #labels = outputs[i*self.batch_size:i*self.batch_size+self.batch_size]
            graph, label = jraph.batch(graphs), np.stack(labels)
            graph = pad_graph_to_nearest_power_of_two(graph)
            self.params, self.opt_state, loss = self.update(self.params, self.opt_state, graph, label)
            loss_sum += loss
        # if number of graphs is not divisible by batch_size, create a batch with leftover graphs
        if total_num_graphs%self.batch_size != 0:
            idx_start = num_training_steps_per_epoch * self.batch_size
            idx_end = total_num_graphs
            graphs = []
            labels = []
            for i in range(idx_start, idx_end):
                graph = inputs[i]
                label = outputs[i]
                graphs.append(graph)
                labels.append([label])
                check_sum += 1
            graph, label = jraph.batch(graphs), np.stack(labels)
            graph = pad_graph_to_nearest_power_of_two(graph)
            self.params, self.opt_state, loss = self.update(self.params, self.opt_state, graph, label)
            loss_sum += loss
            
        if check_sum != total_num_graphs:
            raise RuntimeError('Checksum failed! Graphs expected: {}, graphs used: {}'.format(n_test, check_sum))
        return loss_sum # return the summed loss

    def train(self, train_inputs, train_outputs, epochs):
        '''train the model with training data.'''
        print("starting training \n")
        total_num_graphs = len(train_inputs)
        num_training_steps_per_epoch = total_num_graphs // self.batch_size

        for idx_epoch in range(epochs):
            loss_sum = self.train_epoch(train_inputs, train_outputs, idx_epoch)
            train_inputs, train_outputs = sklearn.utils.shuffle(train_inputs, train_outputs, random_state=0)
            
            print(loss_sum / total_num_graphs)  # print the average loss per graph

    def predict(self, inputs):
        '''Predict outputs based on inputs.'''
        outputs = []
        '''for i in range(len(inputs)):
            if i%1000 == 0:
                print(i)
            pred_graph = self.net.apply(self.params, inputs[i])
            preds = pred_graph.globals
            outputs.append(preds)'''
        graphs = jraph.batch(inputs)
        outputs = self.net.apply(self.params, graphs)

        return outputs.globals

    def test(self, inputs, outputs):
        '''Test the model by evaluating the loss for inputs and outputs. Return MAE.'''
        
        loss_sum = 0.0
        n_test = len(inputs)
        check_sum = 0 # check if all graphs have been used in epoch
        for i in range(n_test // self.batch_size):
            graphs = []
            labels = []
            for idx_batch in range(self.batch_size):
                graph = inputs[i*self.batch_size+idx_batch]
                label = outputs[i*self.batch_size+idx_batch]
                graphs.append(graph)
                labels.append([label])
                check_sum += 1
            # return jraph.batch(graphs), np.concatenate(labels, axis=0)
            graph, label = jraph.batch(graphs), np.stack(labels)
            graph = pad_graph_to_nearest_power_of_two(graph)
            loss, grad = self.compute_loss_fn(self.params, graph, label)
            loss_sum += loss
        if n_test%self.batch_size != 0:
            idx_start = n_test // self.batch_size * self.batch_size
            idx_end = n_test
            graphs = []
            labels = []
            for i in range(idx_start, idx_end):
                graph = inputs[i]
                label = outputs[i]
                graphs.append(graph)
                labels.append([label])
                check_sum += 1
            graph, label = jraph.batch(graphs), np.stack(labels)
            graph = pad_graph_to_nearest_power_of_two(graph)
            loss, grad = self.compute_loss_fn(self.params, graph, label)
            loss_sum += loss

        if check_sum != n_test:
            raise RuntimeError('Checksum failed! Graphs expected: {}, graphs used: {}'.format(n_test, check_sum))
        return loss_sum / n_test

    def train_and_test(self, inputs, outputs, epochs):
        '''Train and validate the model using training data and cross validation.'''
        train_in, test_in, train_out, test_out = sklearn.model_selection.train_test_split(
            inputs, outputs, test_size=0.1, random_state=0
        )
        print("starting training \n")
        total_num_graphs = len(train_in)
        num_training_steps_per_epoch = total_num_graphs // self.batch_size

        for idx_epoch in range(epochs):
            loss_sum = self.train_epoch(train_in, train_out, idx_epoch)
            train_in, train_out = sklearn.utils.shuffle(train_in, train_out, random_state=0)
            
            self.train_loss_arr.append([idx_epoch, loss_sum/total_num_graphs])
            print(loss_sum / total_num_graphs)  # print the average loss per graph

            # every 10 epochs, evaluate test loss
            if idx_epoch%10 == 0:
                test_loss = self.test(test_in, test_out)
                self.test_loss_arr.append([idx_epoch, test_loss])
                print("Test MAE: {}".format(test_loss))
        

    def set_train_logging(self, logging=True):
        self.logging = logging


def main():
    lr = optax.exponential_decay(5*1e-4, 1000, 0.9)
    model = Model(lr, 32, 5)
    file_str = 'aflow/graphs_enthalpy_cutoff4A.csv'
    inputs, outputs = get_data_df_csv(file_str)
    train_in, test_in, train_out, test_out = sklearn.model_selection.train_test_split(
        inputs, outputs, test_size=0.1, random_state=0
    )
    model.build(inputs, outputs)
    # pre training evaluation
    
    preds_train_pre = model.predict(train_in)
    preds_test_pre = model.predict(test_in)
    
    make_result_csv(train_out, preds_train_pre, 'results_test/train_pre.csv')
    make_result_csv(test_out, preds_test_pre, 'results_test/test_pre.csv')
    
    model.train_and_test(inputs, outputs, 50)
    
    # post training evaluation
    preds_train_post = model.predict(train_in)
    preds_test_post = model.predict(test_in)

    
    make_result_csv(train_out, preds_train_post, 'results_test/train_post.csv')
    make_result_csv(test_out, preds_test_post, 'results_test/test_post.csv')

    print(model.train_loss_arr)
    print(model.test_loss_arr)
    np.savetxt("results_test/train_loss_arr.csv", np.array(model.train_loss_arr), delimiter=",")
    np.savetxt("results_test/test_loss_arr.csv", np.array(model.test_loss_arr), delimiter=",")

    

if __name__ == "__main__":
    main()
