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
        
        data.sort(key = lambda row: row[-1])
        self.median_value = data[int(len(data) / 2)][-1]
        self.mean_value = sum(row[-1] for row in data) / len(data)

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
        data.sort(key = lambda row: row[-1])
        self.median_value = data[int(len(data) / 2)][-1]
        self.mean_value = sum(row[-1] for row in data) / len(data)


class DecisionTreeRegressor:
    def __init__(self, random_split=False, max_depth=inf):
        self.random_split = random_split
        self.max_depth = max_depth


    @staticmethod
    def mean_squared_error(data):
        sum_ = sum(row[-1] for row in data)
        mean = sum_ / len(data) if len(data) > 0 else 0

        mse = sum((row[-1] - mean)**2 for row in data)

        return mse / len(data) if len(data) > 0 else 0


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
        
        self.root = self.select_best_feature(data, self.mean_squared_error(data), 0)


    # returns the mse of a split
    # and a dict that contains mse of individual partitions
    def evaluate_split_numerical(self, data, column_id):
        best_split_mse = inf
        best_split_threshold = None
        best_split_partition_mses = None

        # sort rows according to values of the current column
        data.sort(key = lambda row: row[column_id])

        for i, row in enumerate(data):
            if i > 0 and row[column_id] != data[i - 1][column_id]:
                partition_1_mse = DecisionTreeRegressor.mean_squared_error(data[0 : i])
                partition_2_mse = DecisionTreeRegressor.mean_squared_error(data[i : len(data)])

                split_mse = (i * partition_1_mse + (len(data) - i) * partition_2_mse) / len(data)

                if split_mse < best_split_mse:
                    best_split_mse = split_mse
                    best_split_threshold = (row[column_id] + data[i - 1][column_id]) / 2
                    best_split_partition_mses = [partition_1_mse, partition_2_mse]


        return best_split_mse, best_split_partition_mses, best_split_threshold


    # returns the mse of a split
    # and a dict that contains mse of individual partitions
    def evaluate_split_categorical(self, data, column_id):
        col_class_histogram = Counter([row[column_id] for row in data])

        partitions_means = dict.fromkeys(col_class_histogram, 0)
        for row in data:
            partitions_means[row[column_id]] += row[-1]

        partitions_means.update(
            (key, sum_ / col_class_histogram[key]
        ) for key, sum_ in partitions_means.items())

        partitions_mses = dict.fromkeys(col_class_histogram, 0)
        for row in data:
            partitions_mses[row[column_id]] += (row[-1] - partitions_means[row[column_id]])**2

        partitions_mses.update(
            (class_, sum_ / col_class_histogram[class_]) 
                for class_, sum_ in partitions_mses.items() 
                if col_class_histogram[class_] > 0
        )

        split_mse = 0
        for class_ in col_class_histogram:
            number_of_rows = col_class_histogram[class_]
            H = partitions_mses[class_]
            split_mse += H * number_of_rows / len(data)

        return split_mse, partitions_mses, None


    def select_best_feature(self, data, mse_data, depth):
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


        lowest_mse = inf
        best_split_column_id = -1
        best_split_partitions_mses = {}

        for column_id, column in columns.items():
            if not column.used_column:
                split_evaluation = (
                    self.evaluate_split_numerical if column.numerical_attr 
                    else self.evaluate_split_categorical
                )

                split_mse, partitions_mses, threshold = split_evaluation(data, column_id)

                if split_mse < lowest_mse:
                    lowest_mse = split_mse
                    best_split_column_id = column_id
                    best_split_partitions_mses = partitions_mses
                    best_split_threshold = threshold

        if lowest_mse < mse_data:
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
                    best_split_partitions_mses[0], 
                    depth + 1
                )
                child_right = self.select_best_feature(
                    data_partitions[1], 
                    best_split_partitions_mses[1], 
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
                for class_, mse in best_split_partitions_mses.items():
                    data = data_partitions[class_]

                    child = self.select_best_feature(data, mse, depth + 1)

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


    def predict(self, row, median=False):
        current_node = self.root
        while True:
            try:
                child = current_node.get_child(row)
            except:
                child = None

            if type(current_node) is Leaf or child == None:
                return (
                    current_node.median_value if median 
                    else current_node.mean_value
                )

            current_node = child


    #score using coefficient of determination
    def score(self, data_test):
        data_mean = sum(row[-1] for row in data_test) / len(data_test)
        tss = sum((row[-1] - data_mean)**2 for row in data_test)
        rss = sum((row[-1] - self.predict(row))**2 for row in data_test)

        return (1 - rss / tss) if tss > 0 else 0
