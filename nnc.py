import numpy as np

from keras.datasets import cifar10

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, X):
        num_test=X.shape[0]

        Ypred = np.zeros(num_test, self.y_train.dtype)
        for i in range(num_test):

            distances = np.sum(np.abs(self.x_train - X[i,:]), axis = 1)
            min_index = np.argmin(distances)
            Ypred[i] = self.y_train[min_index]

        return Ypred


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32*32*3)


x_test = x_test.reshape(x_test.shape[0], 32*32*3)


classifier = NearestNeighbor()
classifier.train(x_train[:1000,:], y_train[:1000])
y_pred = classifier.predict(x_test[:100])

err = 0
for index, sample in enumerate(y_pred) :
    if sample != y_test[index]:
        err = err + 1
print(err / 1000)
