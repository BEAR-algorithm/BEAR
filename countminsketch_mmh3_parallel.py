import numpy as np
import numpy.random
from collections import Counter
import random
from copy import deepcopy
import itertools
try:
    import mmh3 as mmh3
except ImportError:
    import pymmh3 as mmh3

class CountMinSketch():

    def __init__(self, W_, H_, C_):
        """
            W = Count Sketch width
            H = Number of hashes (height)
            C = Number of classes
		"""
        self.width = W_
        self.height = H_
        self.C = C_

        #self.sketch = np.random.random((C_, H_, W_))
        #self.sketch = self.sketch / np.sum(self.sketch)
        self.sketch = np.zeros((C_, H_, W_))

    def murmur_hash_parallel(self, key_list, hash_ind_list):
        #return list(map(lambda x:(x[1], x[0], mmh3.hash(x[1],x[0])), itertools.product(hash_ind_list,key_list)))
        return list(map(lambda x:mmh3.hash(x[1],x[0]), itertools.product(hash_ind_list,key_list)))

    def add(self, key_list, value_list):
        hash_list_tiled = np.tile(np.ndarray.flatten(np.tile(range(self.height),(np.size(key_list),1)).T),(1,self.C))[0] #seed ind tiled
        murmur_hash_parallel_result = self.murmur_hash_parallel(key_list,range(self.height))
        width_ind_list_tiled = np.tile(np.mod(murmur_hash_parallel_result,self.width),(1,self.C))[0] #mmh3 tiled
        class_list_tiled = np.ndarray.flatten(np.tile(range(self.C),(np.size(key_list)*self.height,1)).T)
        value_list_tiled = np.ndarray.flatten(np.tile(value_list,(self.height,1)).T)
        binary_list_tiled =  np.tile( np.power(-1,np.abs(murmur_hash_parallel_result)),(1,self.C))[0]
        value_list_tiled = np.multiply(value_list_tiled,binary_list_tiled)
        np.add.at(self.sketch, (class_list_tiled,hash_list_tiled,width_ind_list_tiled),value_list_tiled)


    def query(self, key_list):
        hash_list_tiled = np.tile(np.ndarray.flatten(np.tile(range(self.height),(np.size(key_list),1)).T),(1,self.C))[0]
        murmur_hash_parallel_result = self.murmur_hash_parallel(key_list, range(self.height))
        width_ind_list_tiled = np.tile(np.mod(murmur_hash_parallel_result,self.width),(1,self.C))[0]
        class_list_tiled = np.ndarray.flatten(np.tile(range(self.C),(np.size(key_list)*self.height,1)).T)
        binary_list_tiled =  np.tile(np.power(-1,np.abs(murmur_hash_parallel_result)),(1,self.C))[0]
        value_list_tiled = np.multiply(self.sketch[class_list_tiled,hash_list_tiled,width_ind_list_tiled],binary_list_tiled)
        return np.median(value_list_tiled.reshape(self.C,self.height,np.size(key_list)),axis=1).T
