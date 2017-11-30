# notMNIST
This repo contains the files of notMNIST dataset in ubyte format. It has 1000 test and 6000 training examples.

A modified version of tensorflow's mnist data import file is also in this repo named "notMNIST.py".

To use this dataset with the existing MNIST model, place notMNIST.py file in tensorflow/contrib/learn/python/learn/datasets/ directory.
Move the original mnist.py file to some other directory, so that it can be used again later on if needed.
Rename "notMNIST.py" to "mnist.py"
In your code while using "read_data_sets" function specify the path in which notMNIST dataset is stored
      read_data_set("path/to/notMNIST/data", one_hot=True)
