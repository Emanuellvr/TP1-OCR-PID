from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.records import array

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

#train_X = train_X.numpy()

imgplot = plt.imshow(np.array(train_X[0]))
plt.show()