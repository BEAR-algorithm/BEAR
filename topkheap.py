import numpy as np
import numpy.random
from collections import Counter
import random
from copy import deepcopy
import itertools
from heapq import heapify, heappush, heappop, heappushpop


class mytopkheap():

    def __init__(self, k):

        self.k_ = k
        self.topk_heap = list(zip([0],[0])) ### value, key
        heapify(self.topk_heap)

    def output_topk_heap(self):
        return self.topk_heap

    def push(self, item_list):
        ## here i am doing an approximation, the line below could be inside the
        ## loop, however it would be much much slower.

        topk_heap_index = list(zip(*self.topk_heap))[1]
        topk_heap_index_set = set(list(zip(*self.topk_heap))[1])
        for item in item_list:
            value = item[0]
            key = item[1]

            ## a slower but more accurate way of doing this to comment out the next line
            #topk_heap_index = list(zip(*self.topk_heap))[1]
            #topk_heap_index = list(zip(*self.topk_heap))[1]
            #topk_heap_index_set = set(list(zip(*self.topk_heap))[1])
            if key in topk_heap_index_set:
                heap_index = topk_heap_index.index(key)
                self.topk_heap[heap_index] = item
                heapify(self.topk_heap)

            elif len(self.topk_heap) < self.k_:
                heappush(self.topk_heap,item)

            elif value > self.topk_heap[0][0]:
                popeditem = heappushpop(self.topk_heap,item)
