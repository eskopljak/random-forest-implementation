from math import inf

from DecisionTree.Utils.Read import read_csv
from DecisionTree.Models.DecisionTreeRegressor import DecisionTreeRegressor
from DecisionTree.Utils.Output import print_tree
from DecisionTree.Utils.FilterColumns import filter_columns
from DecisionTree.Utils.TrainTestSplit import train_test_split

from print_exectime import print_exectime


columns, data = print_exectime(lambda: read_csv('TestData/MallCustomers.csv'), 'Loading data')

columns, data = print_exectime(lambda: filter_columns(columns, data, range(1, 5)), 'Filtering columns')

data_train, data_test = train_test_split(data, train_size = 0.7)

tree = DecisionTreeRegressor(max_depth=inf)
print_exectime(lambda: tree.fit(columns, data_train), 'Tree induction')

print(f'Score: {tree.score(data_test)}')

with open('Output/tree.txt', 'w') as f:
    print_exectime(lambda: print_tree(tree, file=f), 'Output')
