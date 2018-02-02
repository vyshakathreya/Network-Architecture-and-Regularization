# Network-Architecture-and-Regularization
Exploration of different neural network architectures. 

Data set used is MIT/TIDIGITS corpus (Leonard and Doddington, 1993). 

key features include:
1. Automatic caching of the PCA analysis and feature generation
2. A data structure (array) that holds keras network architectures
3. Cross validation saves results from each fold. Results are accessible by getMethods()

Different network architectures and regularizers: L1, L2, and dropout are explored here. 

Best architecture found is 20 input layers, 2 hidden layers of 20 and 40 nodes with L2 regularizer
