from math import inf

from DecisionTree.Utils.Read import read_csv
from DecisionTree.Models.RandomForestRegressor import RandomForestRegressor
from DecisionTree.Utils.Output import print_tree
from DecisionTree.Utils.FilterColumns import filter_columns
from DecisionTree.Utils.TrainTestSplit import train_test_split

from print_exectime import print_exectime


if __name__ == '__main__':
    columns, data = print_exectime(lambda: read_csv('TestData/MallCustomers.csv'), 'Loading data')
    
    columns, data = print_exectime(lambda: filter_columns(columns, data, range(1, 5)), 'Filtering columns')
    
    data_train, data_test = train_test_split(data, train_size = 0.7)
    
    forest = RandomForestRegressor(n_trees=10, max_depth=inf)
    
    print_exectime(lambda: forest.fit(columns, data_train), 'Forest induction')
    
    print(f'Score: {forest.score(data_test)}')

    for i, tree in enumerate(forest.trees):
            with open(f'Output/tree{i + 1}.txt', 'w') as f:
                print_tree(tree, file=f)
