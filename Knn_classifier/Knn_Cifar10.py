import cifar10.cifar10_handler as cifar10
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy.linalg as alg


def l1_distance(data, test_point):
    return np.sum(np.absolute(data - test_point), axis=1)
    # return alg.norm(distance,ord=1)


def l1_distance_cdist(train_data, test_data):
    return cdist(train_data, test_data)


class Knn_classifier(object):
    def __init__(self, train_data, train_labels, test_data, test_labels, k):
        self.train_data = train_data  # data will be list of dicts (output off handler.get_cifar_data)
        self.train_labels = train_labels  # data will be list of dicts (output off handler.get_cifar_data)
        self.test_data = test_data  # data will be list of dicts (output off handler.get_cifar_data)
        self.test_labels = test_labels  # data will be list of dicts (output off handler.get_cifar_data)
        self.k = k

    def neighbors_ids(self, distances):
        return np.argpartition(distances, self.k)[:self.k]

    def neighbors_labels(self, neighbors_ids):
        return self.train_labels[neighbors_ids]

    def get_prediction(self, neigbors_labels):
        return neigbors_labels.max()

    def get_how_many_equal(self, a, b):
        count = 0
        for num in range(min(len(a), len(b))):
            if a[num] == b[num]:
                count += 1
        return count

    def get_test_accuracy_cdist(self):
        distances = l1_distance_cdist(self.train_data, self.test_data[:500])
        idx = np.argpartition(distances, self.k, axis=0)[:self.k]
        nearest_dists = np.take(self.train_labels, idx)
        predictions = np.squeeze(mode(nearest_dists, axis=0)[0])
        amount_correct = self.get_how_many_equal(predictions, self.test_labels)
        return amount_correct / float(len(predictions))

    def predict(self, test_point):
        distances = l1_distance(self.train_data, test_point)
        k_neighbours = self.neighbors_ids(distances)
        neigbors_labels = self.neighbors_labels(k_neighbours)
        return self.get_prediction(neigbors_labels)

    def get_test_accuracy(self):
        count = 0
        pbar = tqdm(total=len(self.test_labels))
        for (test_point, label) in zip(self.test_data, self.test_labels):
            pbar.update(1)
            prediction = self.predict(test_point)
            if prediction == label:
                count += 1
        return count / len(self.test_labels)


if __name__ == '__main__':
    cifar10_handler = cifar10.Handler('./cifar10/data', 'data_batch_', range(1, 6))
    cifar_dict = cifar10_handler.get_cifar_dict()
    data = cifar_dict['data']
    labels = np.array(cifar_dict['labels'])
    cifar10_test = cifar10_handler.get_test()
    classifier = Knn_classifier(data, labels, cifar10_test['data'], cifar10_test['labels'], k=6)

    print(classifier.get_test_accuracy())
    # print(cifar10_test['labels'][0])
