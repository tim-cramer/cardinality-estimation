import math
import heapq
from abc import ABC, abstractmethod

from src.utils import distinct_hash


class CardinalityEstimator(ABC):
    @abstractmethod
    def add(self, item):
        pass

    @abstractmethod
    def estimate(self):
        pass
    
    @property
    def name(self):
        return self.__class__.__name__

class TrueCardinality(CardinalityEstimator):
    def __init__(self):
        self.seen = set()
    def add(self, item):
        self.seen.add(item)
    def estimate(self):
        return len(self.seen)

class HyperLogLog(CardinalityEstimator):
    def __init__(self, b=10, seed=0):
        self.b = b
        self.seed = seed
        self.m = 1 << b
        self.M = [0] * self.m
        # Alpha constants
        if self.m == 16: self.alpha = 0.673
        elif self.m == 32: self.alpha = 0.697
        elif self.m == 64: self.alpha = 0.709
        else: self.alpha = 0.7213 / (1 + 1.079 / self.m)

    @property
    def name(self):
        return f"HLL (b={self.b})"

    def add(self, item):
        x = distinct_hash(item, self.seed)
        j = x & (self.m - 1)
        w = x >> self.b
        rho = (w & -w).bit_length() if w != 0 else 32 - self.b + 1
        self.M[j] = max(self.M[j], rho)

    def estimate(self):
        z_inv = sum(2**-reg for reg in self.M)
        E = self.alpha * (self.m ** 2) / z_inv
        if E <= 2.5 * self.m:
            V = self.M.count(0)
            if V != 0: E = self.m * math.log(self.m / V)
        return E

class Recordinality(CardinalityEstimator):
    def __init__(self, k=64, use_hash=True, seed=0):
        self.k = k
        self.seed = seed
        self.sample = [] 
        self.R = 0 
        self.use_hash = use_hash

    @property
    def name(self):
        h_str = "Hash" if self.use_hash else "NoHash"
        return f"REC (k={self.k}, {h_str})"

    def add(self, item):
        if self.use_hash:
            val = distinct_hash(item, self.seed)
        else:
            try:
                val = int(item) 
            except ValueError:
                val = distinct_hash(item, self.seed)

        if len(self.sample) < self.k:
            if val not in self.sample:
                heapq.heappush(self.sample, val)
                self.R += 1
        else:
            if val > self.sample[0] and val not in self.sample:
                heapq.heappop(self.sample)
                heapq.heappush(self.sample, val)
                self.R += 1

    def estimate(self):
        term = (1 + 1/self.k) ** (self.R - self.k + 1)
        return self.k * term - 1

class PCSA(CardinalityEstimator):
    def __init__(self, b=10, seed=0):
        self.b = b  
        self.seed = seed
        self.m = 1 << b 
        self.bitmaps = [0] * self.m
        self.phi = 0.77351 

    @property
    def name(self):
        return f"PCSA (b={self.b})"

    def add(self, item):
        x = distinct_hash(item, self.seed)
        j = x & (self.m - 1)
        w = x >> self.b
        if w == 0:
            rho = 32 - self.b
        else:
            rho = (w & -w).bit_length() - 1 
        
        self.bitmaps[j] |= (1 << rho)

    def estimate(self):
        S = 0
        for map_val in self.bitmaps:
            # Find first zero in the bitmap
            if map_val == 0:
                R = 0
            else:
                R = (~map_val & (map_val + 1)).bit_length() - 1
            S += R
        
        return self.m * (2 ** (S / self.m)) / self.phi