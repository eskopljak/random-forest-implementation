from collections import Counter
from math import inf
from random import random
from multiprocessing import Process, Queue

from DecisionTree.Models.DecisionTreeClassifier import DecisionTreeClassifier
from DecisionTree.Utils.BootstrapSample import bootstrap_sample


class ProcessCTree(Process):
    def __init__(self, queue, columns, data, criterion, max_depth):
        super(ProcessCTree, self).__init__()
        self.queue = queue
        self.columns = columns
        self.data = data
        self.criterion = criterion
        self.max_depth = max_depth
        

    def run(self):
        tree = DecisionTreeClassifier(
            random_split = True, 
            criterion = self.criterion, 
            max_depth = self.max_depth
        )
        tree.fit(self.columns, self.data)
        self.queue.put(tree)


class RandomForestClassifier:
    def __init__(self,  n_trees = 100, criterion = 'entropy', max_depth = inf):
        self.trees = []
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth


    def fit(self, columns, data):
        queue = Queue()
        processes = []
        for _ in range(self.n_trees):
            p = ProcessCTree(
                queue=queue, 
                columns=columns, 
                data=bootstrap_sample(data, len(data)), 
                criterion=self.criterion, 
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
        predictions = Counter()
        for tree in self.trees:
            predictions[tree.predict(row)] += 1

        return predictions.most_common(1)[0][0]


    def score(self, data_test):
        sum_ = sum((1 if self.predict(row) != row[-1] else 0 for row in data_test))
        return 1 - sum_ / len(data_test)
