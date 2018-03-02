import cPickle
import os
import numpy as np


class Handler(object):
    def __init__(self,data_dir='./data',data_name_moc='data_batch_',num_range=range(1,6)):
        self.data_dir=data_dir
        self.data_name_moc=data_name_moc
        self.num_range=num_range

    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = cPickle.load(fo)
        return dict


    def get_cifar_data(self):
        cifar_data=[]
        for num in self.num_range:
            mini_batch=self.unpickle(os.path.join(self.data_dir,self.data_name_moc+str(num)))
            cifar_data.append(mini_batch)
        return cifar_data

    def get_test(self):
        return self.unpickle(os.path.join(self.data_dir,'test_batch'))

    def get_cifar_dict(self):
        cifar_data={}
        for num in self.num_range:
            mini_batch=self.unpickle(os.path.join(self.data_dir,self.data_name_moc+str(num)))
            if 'data' in cifar_data:
                cifar_data['data']=np.append(cifar_data['data'],mini_batch['data'],axis=0)
                cifar_data['labels'] += mini_batch['labels']
                cifar_data['filenames'] += mini_batch['filenames']
            else:
                cifar_data['data']=mini_batch['data']
                cifar_data['labels']=mini_batch['labels']
                cifar_data['filenames']=mini_batch['filenames']



        return cifar_data
