from __future__ import division

from bitarray import bitarray
from collections import Counter
import numpy as np
import hashlib
import sys
import math
import pickle

class bloomFilter:
    
    def __init__(self, N, p):
        self.m = self.__roundBase8(-1*N*math.log(p)/(math.log(2)**2))
        self.barray = bitarray(self.m)
        self.barray.setall(0)
    
    def __roundBase8(self, x):
        return int(8 * round(float(x)/8))
    
    def __rehash(self, x, h):
        return (x^h) % self.m
        
    def fill(self, x, hashes):
        for h in hashes:
            self.barray[self.__rehash(x, h)] = 1
    
    def query(self, x, hashes):
        for h in hashes:
            if not self.barray[self.__rehash(x, h)]:
                return False
        return True
    
class Node:
    
    def __init__(self, values=None, hashes=None, child_left=None, child_right=None):
        self.values = values
        self.hashes = hashes
        self.child_left = child_left
        self.child_right = child_right

class bloomTree:
    
    def __init__(self, P=None):
        self.P = P
    
    def __hash(self, x, nonce=''):
        return int(hashlib.md5(str(x) + str(nonce)).hexdigest(), 16)
        
    def __hash_generator(self, k=1, seed=0):
        while True:
            H = []
            for _ in range(k):
                seed += math.pi
                H.append(self.__hash(seed))
            yield H
            
    def __build(self, values, hist):
        
        #Initialize tree and stack 
        self.head_node = Node(values)
        stack = [self.head_node]
        
        # Initialize running count of encoded elements 
        N_R = 0
        N_L = 0
        
        # Generate Bloom Tree via DFS 
        while stack:
            node = stack.pop()
            
            # Base case
            if len(node.values) == 1:
                continue
            
            # Each node splits the values in half
            half = len(node.values)//2
            node_l = Node(node.values[:half])
            node_r = Node(node.values[half:])
            
            # Count elements encoded (right) and not encoded (left) in BF 
            N_R += sum(map(lambda x: hist[x], node_r.values))
            N_L += sum(map(lambda x: hist[x], node_l.values))
            
            # Grow tree
            if node_r.values:
                node.child_right = node_r
                stack.append(node_r)
            if node_l.values:
                node.child_left = node_l
                stack.append(node_l)
                
        return N_R, N_L
    
    def __average_size_of_hashtable(self, N):
        """ Approx average size for int keys/values"""
        load_factor = (3./2. + 2.)/2.
        m_key = 24.
        m_value = 24.
        return load_factor*(m_key + m_value)*N
                
    def __query_bloom_tree(self, x):
        node = self.head_node
        hashed_key = self.__hash(x)
        while len(node.values) > 1:
            in_right_set = self.bloom_filter.query(hashed_key, node.hashes)
            node = node.child_right if in_right_set else node.child_left
        return node.values[0]
        
    def fill(self, hashtable):
        
        hist = Counter(hashtable.values())
        values = zip(*hist.most_common())[0] 
        N = len(hashtable)
        
        # Build tree
        self.N_R, self.N_L = self.__build(values, hist)
        self.R = self.N_L/(self.N_R + self.N_L)
        
        # Calculate parameters
        self.B = len(hist)
        if self.P:
            self.P_hat = self.P
        else:
            #self.P_hat = 1./(math.log(2)**2)*self.N_R/(8*self.__average_size_of_hashtable(N))
            self.P_hat = 1./(math.log(2)**2)*self.N_R/(8*sys.getsizeof(hashtable))
        self.p_fp = -1.0/(self.R*math.log(self.B,2))*math.log1p(-self.P_hat)
        self.k = int(math.ceil(-math.log(self.p_fp, 2)))
        
        # Initialize bloom filter
        self.bloom_filter = bloomFilter(self.N_R, self.p_fp)
        
        # Initialize hash generator 
        hasher = self.__hash_generator(self.k)
        
        #Initialize stack
        stack = [self.head_node]
    
        # Fill bloom filter by traversing through tree via DFS
        while stack:
            node = stack.pop()
            
            # Base case
            if len(node.values) == 1:
                continue
            
            # Generate hashes 
            node.hashes = hasher.next()
            
            # Only need to encode keys associated with right split
            keys_to_encode = filter(lambda x: hashtable[x] in node.child_right.values, hashtable.keys())
            hashed_keys = map(self.__hash, keys_to_encode)
            map(lambda x: self.bloom_filter.fill(x, node.hashes), hashed_keys)
            
            # Next level
            stack += [node.child_right, node.child_left]
        
        # find all false positives and add to false positive dictionary
        self.fp_hashtable = {}
        mcr = 0
        for k, v in hashtable.items():
            if self.__query_bloom_tree(k) != v:
                self.fp_hashtable[k] = v
                mcr += 1
        self.mcr = mcr/float(N)
    
    def query(self, x):
        return self.fp_hashtable[x] if x in self.fp_hashtable else self.__query_bloom_tree(x)
    
    def size(self, of='total', average=True):
        if of == 'total':
            if average:
                return (self.bloom_filter.barray.buffer_info()[4] 
                        + self.__average_size_of_hashtable(len(self.fp_hashtable)))
            else:
                return (self.bloom_filter.barray.buffer_info()[4] 
                        + sys.getsizeof(self.fp_hashtable))
        
        if of == 'hashtable':
            if average:
                return self.__average_size_of_hashtable(len(self.fp_hashtable))
            else:
                 return sys.getsizeof(self.fp_hashtable) 
                                                        
        if of == 'bloomtree':
            return self.bloom_filter.barray.buffer_info()[4] 
        
    def bt_fpr(self):
        return self.fpr
        
    def bt_params(self):
        print "Bloom Tree parameters"
        print "----------------------"
        print "Number of elements inserted into Bloom filter:", self.N_R
        print "Prob. of false positve for Bloom filter:", self.p_fp
        print "Number of buckests/values:", self.B
        print "Number of hash functions per node:", self.k
        print "Theoretical miss-classification prob.", self.P_hat
        print "Total number of bits needed for Bloom filter:", self.bloom_filter.m
        print "Empirical miss-classification rate:", self.mcr
        
    def bf_params(self):
        params = self.bloom_filter.barray.buffer_info()
        print "Bloom Filter parameters"
        print "-----------------------"
        print "Size (bytes):", params[1]
        print "Size (bits):", params[1]*8
        print "Unused bits:", params[3]
        print "Allocated memory (bytes):", params[4]