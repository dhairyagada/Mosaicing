import tensorflow as tf
import numpy as np

X = np.load('TestDataX.npy')

U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
print("X:", X.shape)
print("U:", U.shape)
print("Sigma:", Sigma.shape)
print("V^T:", VT.shape)

num_components = 8 # Number of principal components
Y = np.matmul(X, VT[:num_components,:].T)
print(X)
print(Y)