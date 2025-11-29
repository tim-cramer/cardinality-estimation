import numpy as np

def generate_zipf_stream(n, alpha, N):
    """
    Generates a stream of N items from a universe of n distinct elements
    following a Zipfian distribution with parameter alpha.
    """
    if alpha <= 0:
        raise ValueError("Alpha must be >= 0")
    
    ranks = np.arange(1, n + 1)
    probabilities = 1 / (ranks ** alpha)
    probabilities /= probabilities.sum() # Normalize to c_n
    
    stream = np.random.choice(ranks, size=N, p=probabilities)
    return stream.astype(str) 