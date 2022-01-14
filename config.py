'''Config module for global variables. This is the (canonical) way.'''

LABEL_SIZE = None # must define before use, no default
N_HIDDEN_C = 32
MAX_ATOMIC_NUMBER = 100
AVG_MESSAGE = False # if the average instead of sum should be used for message aggregation
AVG_READOUT = False # if the average instead of sum should be used for readout function
NUM_MP_LAYERS = 3 # number of message passing layers

HK_INIT = None