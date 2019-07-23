from math import inf
from random import random
from multiprocessing import Process, Queue

from DecisionTree.Models.DecisionTreeRegressor import DecisionTreeRegressor
from DecisionTree.Utils.BootstrapSample import bootstrap_sample


class ProcessRTree(Process):
    def __init__(self, queue, columns, data, max_depth):
        super(ProcessRTree, self).__init__()
        self.queue = queue
        self.columns = columns
        self.data = data
        self.max_depth = max_depth


    def run(self):
        tree = DecisionTreeRegressor(
            random_split = True, 
            max_depth = self.max_depth
        )
        tree.fit(self.columns, self.data)
        self.queue.put(tree)


class RandomForestRegressor:
    def __init__(self,  n_trees = 100, max_depth = inf):
        self.trees = []
        self.n_trees = n_trees
        self.max_depth = max_depth


    def fit(self, columns, data):
        queue = Queue()
        processes = []
        for _ in range(self.n_trees):
            p = ProcessRTree(
                queue=queue, 
                columns=columns, 
                data=bootstrap_sample(data, len(data)), 
                max_depth=self.max_depth
            )
            p.daemon = True
            p.start()
            processes.append(p)

        for _ in range(self.n_trees):
            new_tree = queue.get()
            self.trees.append(new_tree)
            
        for p in processes:
            p.join()


    def predict(self, row):
        predictions_sum = 0
        for tree in self.trees:
            predictions_sum += tree.predict(row)

        return predictions_sum / len(self.trees)


    #score using coefficient of determination
    def score(self, data_test):
        data_mean = sum(row[-1] for row in data_test) / len(data_test)
        tss = sum((row[-1] - data_mean)**2 for row in data_test)
        rss = sum((row[-1] - self.predict(row))**2 for row in data_test)

        return (1 - rss / tss) if tss > 0 else 0
