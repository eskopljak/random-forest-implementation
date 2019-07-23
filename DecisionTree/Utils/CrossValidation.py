from random import shuffle
from copy import copy

def create_partitions(data, n):
    data = copy(data)
    shuffle(data)

    size = int(len(data) / n)
    partitions = [data[i * size : (i + 1) * size] for i in range(n)]

    return partitions    


def cv_score(model, columns, partitions):
    total_error = 0
    for i in range(len(partitions)):
        data_test = partitions[i]

        data_train = sum((partition for j, partition in enumerate(partitions) if j != i), [])

        model.fit(columns, data_train)

        mean_error = model.mean_error(data_test)
        total_error += mean_error

    return total_error / len(partitions)
