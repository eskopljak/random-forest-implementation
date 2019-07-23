from random import shuffle
from copy import copy

def train_test_split(data, train_size):
    data = copy(data)
    shuffle(data)

    boundary = int(train_size * len(data))
    return data[0 : boundary], data[boundary : len(data)]