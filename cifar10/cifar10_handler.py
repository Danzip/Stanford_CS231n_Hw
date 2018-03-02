import cPickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def get_cifar_data(data_dir='./data',data_name_moc='data_batch_',num_range=range(1,6)):
    cifar_data={}
    for num in num_range:
        mini_batch=unpickle(os.path.join(data_dir,data_name_moc+str(num)))
        cifar_data.update(mini_batch)
    return cifar_data