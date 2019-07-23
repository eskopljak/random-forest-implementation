from random import random
from copy import copy

def bootstrap_sample(data, sample_size, with_replacement = True):
    if not with_replacement:
        data = copy(data)

    sample = []
    for _ in range(sample_size):
        if len(data) == 0:
            break

        index = int(random() * len(data))
        sample.append(data[index])

        if not with_replacement:
            data.pop(index)
    
    return sample