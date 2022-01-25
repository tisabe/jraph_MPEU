# jraph_MPEU
Message Passing Graph Neural Network with Edge Updates in Jraph

This code implements a graph neural network with the architecture described in
https://arxiv.org/pdf/1806.03146.pdf
"Neural Message Passing with Edge Updates for
Predicting Properties of Molecules and Materials"

All this code is experimental, run at your own risk!

Only tested on Python 3.7.3

Python library requirements:
Jax (CPU or GPU)
tensorflow
spektral
numpy
jraph
haiku
optax
pandas
tqdm
sklearn
pickle

If any other libraries are missing, just pip install them, the above list might not be complete.

How to get this running:
At the moment this can be run with two different datasets: QM9 and aflow.

The QM9 dataset is pulled from spektral and converted into graphs.
To get the QM9 dataset run datahandler_QM9.py. You might have to specify and make output file by hand.

The aflow dataset is just a small testset of materials pulled directly with the alfow API with a json response.
To do this first run datapuller.py, you may have to make a directory "aflow".
Then run datahandler.py to convert the raw data from the pull into graphs.

To train a model run model.py, this defaults to the QM9 dataset, but does not automatically pull it.


