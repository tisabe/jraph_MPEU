# jraph_MPEU
Message Passing Graph Neural Network with Edge Updates in Jraph.

This code implements a graph neural network with the architecture described in
https://arxiv.org/pdf/1806.03146.pdf
"Neural Message Passing with Edge Updates for
Predicting Properties of Molecules and Materials"

All this code is experimental, run at your own risk!

Only tested on Python 3.7.3

## Python library requirements:  
Jax (only GPU version has been tested)  
tensorflow  
spektral  
numpy  
flax  
jraph  
haiku  
optax  
pandas  
tqdm  
sklearn  
pickle  
ase  

If any other libraries are missing, just pip install them, the above list might not be complete.

We use:
- Optax for the training optimizer.
- Jraph for the graph neural network.
- Haiku for the fully connected neural networks (used to compute edge/message updates and for the readout function).
- Flax for the training loop (it keeps track of the training state and works well with Optax).


## How to get this running:  
At the moment this can be run with two different datasets: QM9 and aflow.

## Datasets
The QM9 dataset is pulled from spektral and converted into graphs.
To get the QM9 dataset run datahandler_QM9.py. You might have to specify and make output file by hand.

The aflow dataset is just a small testset of materials pulled directly with the alfow API with a json response.
To do this first run datapuller.py, you may have to make a directory "aflow".
Then run datahandler.py to convert the raw data from the pull into graphs.

## Training
To train a model run main.py, this defaults to the QM9 dataset, but does not automatically pull it. You have to specify config directory, where parameters are pulled and working directory, where results are stored.

The config directory can be one of the two files in configs.

## Hardware:  
Only validated on NVIDIA Quadro RTX 4000 with 8GB of VRAM. Quadro P400 with 2GB of VRAM runs out of memory.



