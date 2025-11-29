import mmh3

def distinct_hash(item, seed=0):
    return mmh3.hash(str(item), seed, signed=False)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().split()