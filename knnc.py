import numpy as np
import math

from keras.datasets import cifar10

class KNearestNeighbor(object):
    def __init__(self, nbNeighbors):
        self.k = nbNeighbors
        pass

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train


    def getNeighbors(self, X):
        neighbors = []
        for testSample in range(len(X)):
            neighbors.append([])
            dist = np.sum(np.abs(self.x_train - X[testSample, :]), axis = 1)
            for x in range(self.k):
                min_index = np.argmin(dist)
                neighbors[testSample].append(min_index)
                dist[min_index] = np.max(dist)
        self.neighbors = np.array(neighbors)

    def getResponse(self):
        votes = {}
        response = []
        for testItem in self.neighbors:
            votes = {}
            for x in testItem:
                if self.y_train[x][0] in votes:
                    votes[self.y_train[x][0]] = votes[self.y_train[x][0]] + 1
                else:
                    votes[self.y_train[x][0]] = 1
            response.append(max(votes, key=votes.get))
        return response


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)


classifier = KNearestNeighbor(3)
classifier.train(x_train[:1000,:], y_train[:1000])
classifier.getNeighbors(x_test[:100, :])
y_pred = classifier.getResponse()

err = 0
for index, sample in enumerate(y_pred) :
    if sample != y_test[index]:
        err = err + 1
print(err / 1000)
