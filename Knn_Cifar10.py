import cifar10.cifar10_handler as handler
import os


def knn_classifier():
    data=handler.get_cifar_data('./cifar10/data','data_batch_',range(1,6))
    pass

if __name__=='__main__':
    knn_classifier()