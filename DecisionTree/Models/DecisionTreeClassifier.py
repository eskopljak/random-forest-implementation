from collections import Counter, defaultdict
from math import log, inf, sqrt

from DecisionTree.Utils.BootstrapSample import bootstrap_sample


class Column:
    def __init__(self, column_name, numerical_attr, used_column):
        self.column_name = column_name
        self.numerical_attr = numerical_attr
        self.used_column = used_column


class NodeCategorical:
    def __init__(self, column_id, children, data):
        self.column_id = column_id
        # children => dict: {'Sunny': &Node1, 'Cloudy': &Node2...}
        self.children = children

        histogram = Counter(row[-1] for row in data)
        self.most_common_cls = histogram.most_common(1)[0][0]

    def get_child(self, row):
        value = row[self.column_id]
        return self.children.get(value, None)


class NodeNumerical:
    def __init__(self, column_id, children, threshold):
        self.column_id = column_id
        self.threshold = threshold
        # children => list: [&Node1, &Node2...]
        self.children = children

    def get_child(self, row):
        value = row[self.column_id]        
        return self.children[0 if value <= self.threshold else 1]


class Leaf:
    def __init__(self, data):
        histogram = Counter(row[-1] for row in data)

        self.histogram = histogram
        self.most_common_cls = histogram.most_common(1)[0][0]


class DecisionTreeClassifier:
    def __init__(self, criterion = 'entropy', random_split = False, max_depth = inf):
        if criterion == 'entropy':
            self.metrics = DecisionTreeClassifier.entropy
        else:
            self.metrics = DecisionTreeClassifier.gini_impurity

        self.random_split = random_split
        self.max_depth = max_depth


    @staticmethod
    def entropy(histogram, number_of_rows):
        return -sum(p * log(p, 2) for p in (x / number_of_rows for x in histogram) if p > 0)


    @staticmethod
    def gini_impurity(histogram, number_of_rows):
        return 1 - sum((x / number_of_rows)**2 for x in histogram)


    def fit(self, columns, data):
        self.columns = dict.fromkeys(range(len(columns)))

        # every column whose name starts with 'n' or 'N' represents a numerical feature
        # last column represents the target feature
        self.columns.update(
            (index, Column(
                    column_name = column_name[1:],
                    numerical_attr = (
                        column_name.startswith('n') or column_name.startswith('N')
                    ), 
                    used_column = (
                        False if index < len(columns) - 1 
                        else True
                    )
                )
            ) 
            for index, column_name in enumerate(columns))
        
        target_values = [row[-1] for row in data]
        self.target_values = set(target_values)

        number_of_rows = len(data)
        
        histogram = Counter(target_values)
        
        self.root = self.select_best_feature(
            data, 
            self.metrics(histogram.values(), number_of_rows), 
            0
        )


    # returns the entropy of a split
    # and a dict that contains entropies of individual partitions
    def evaluate_split_numerical(self, data, column_id):
        best_split_entropy = inf
        best_split_threshold = None
        best_split_partition_entropies = None

        # sort rows according to values of the current column
        data.sort(key = lambda row: row[column_id])

        target_values_rows_left = Counter([row[-1] for row in data])
        target_values_rows_so_far = dict.fromkeys(target_values_rows_left, 0)

        if len(data) > 0:
            previous_class = data[0][-1]
        for i, row in enumerate(data):
            if row[-1] != previous_class and row[column_id] != data[i - 1][column_id]:
                # boundary between two classes, evaluate split
                partition_1_entropy = self.metrics(target_values_rows_so_far.values(), i) 
                partition_2_entropy = self.metrics(target_values_rows_left.values(), len(data) - i)

                split_entropy = (i * partition_1_entropy + (len(data) - i) * partition_2_entropy) / len(data)

                if split_entropy < best_split_entropy:
                    best_split_entropy = split_entropy
                    best_split_threshold = (row[column_id] + data[i - 1][column_id]) / 2
                    best_split_partition_entropies = [partition_1_entropy, partition_2_entropy]

                previous_class = row[-1]
            
            target_values_rows_left[row[-1]] -= 1
            target_values_rows_so_far[row[-1]] += 1

        return best_split_entropy, best_split_partition_entropies, best_split_threshold


    # returns the entropy of a split
    # and a dict that contains entropies of individual partitions
    def evaluate_split_categorical(self, data, column_id):
        col_class_histogram = Counter([row[column_id] for row in data])

        # dict where key={col_class_name} and value is a new dict
        # which represents a histogram for target class values in a given partition
        dict_partitions_info = dict.fromkeys(col_class_histogram)

        for key in dict_partitions_info:
            dict_partitions_info[key] = dict.fromkeys(self.target_values, 0)

        for row in data:
            dict_histogram = dict_partitions_info[row[column_id]]
            dict_histogram[row[-1]] += 1

        split_entropy = 0
        partitions_entropy = dict.fromkeys(col_class_histogram)
        for class_, dict_histogram in dict_partitions_info.items():
            number_of_rows = col_class_histogram[class_]

            H = self.metrics(dict_histogram.values(), number_of_rows)

            partitions_entropy[class_] = H

            split_entropy += H * number_of_rows / len(data)

        return split_entropy, partitions_entropy, None


    def select_best_feature(self, data, entropy_data, depth):
        if depth >= self.max_depth:
            return Leaf(data)

        columns = self.columns
        if self.random_split:
            # random subset of columns is considered for a split
            column_indices = []
            for column_id, column in self.columns.items():
                if not column.used_column:
                    column_indices.append(column_id)

            column_indices = bootstrap_sample(
                column_indices, 
                int(sqrt(len(self.columns))), 
                with_replacement=False
            )

            columns = {}
            for i in column_indices:
                columns[i] = self.columns[i]


        lowest_entropy = inf
        best_split_column_id = -1
        best_split_partitions_entropies = {}

        for column_id, column in columns.items():
            if not column.used_column:
                split_evaluation = (
                    self.evaluate_split_numerical if column.numerical_attr 
                    else self.evaluate_split_categorical
                )

                split_entropy, partitions_entropies, threshold = split_evaluation(data, column_id)
                
                if split_entropy < lowest_entropy:
                    lowest_entropy = split_entropy
                    best_split_column_id = column_id
                    best_split_partitions_entropies = partitions_entropies
                    best_split_threshold = threshold

        if lowest_entropy < entropy_data:
            # split
            chosen_column = self.columns[best_split_column_id]                

            if chosen_column.numerical_attr:
                data_partitions = [[], []]
                for row in data:
                    index = (
                        0 if row[best_split_column_id] <= best_split_threshold 
                        else 1
                    )
                    data_partitions[index].append(row)

                child_left = self.select_best_feature(
                    data_partitions[0], 
                    best_split_partitions_entropies[0], 
                    depth + 1
                )
                child_right = self.select_best_feature(
                    data_partitions[1], 
                    best_split_partitions_entropies[1], 
                    depth + 1
                )

                return NodeNumerical(
                    best_split_column_id, 
                    [child_left, child_right], 
                    best_split_threshold
                )
            
            else:
                chosen_column.used_column = True

                data_partitions = defaultdict(list)
                for row in data:
                    key = row[best_split_column_id]
                    data_partitions[key].append(row)

                new_children = {}
                for class_, entropy in best_split_partitions_entropies.items():
                    data = data_partitions[class_]

                    child = self.select_best_feature(data, entropy, depth + 1)

                    new_children[class_] = child

                chosen_column.used_column = False

                return NodeCategorical(
                    best_split_column_id, 
                    new_children, 
                    data
                )

        else:
            # no splitting, leaf
            return Leaf(data)
            

    def predict(self, row):
        current_node = self.root
        while True:
            try:
                child = current_node.get_child(row)
            except:
                child = None
            
            if type(current_node) is Leaf or child == None:
                return current_node.most_common_cls

            current_node = child


    def score(self, data_test):
        sum_ = sum((1 if self.predict(row) != row[-1] else 0 for row in data_test))
        return 1 - sum_ / len(data_test)
