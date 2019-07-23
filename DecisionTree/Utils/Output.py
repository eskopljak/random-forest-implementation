from sys import stdout

from DecisionTree.Models.DecisionTreeClassifier import Leaf as LeafClassifier
from DecisionTree.Models.DecisionTreeRegressor import Leaf as LeafRegressor


def __print_tree(output, node, columns, indentation):
    if type(node) is LeafClassifier:
        print(f'{indentation}--> {node.histogram}', file=output)
    elif type(node) is LeafRegressor:
        print(f'{indentation}--> Mean value: {node.mean_value}', file=output)
    else:
        column = columns[node.column_id]
        print(f'{indentation}<<< {column.column_name} >>>', file=output)
        indentation += '        '
        if column.numerical_attr:
            print(f'{indentation}[{column.column_name} <= {node.threshold}]', file=output)
            __print_tree(output, node.children[0], columns, indentation + '        ')
            print(f'{indentation}[{column.column_name} > {node.threshold}]', file=output)
            __print_tree(output, node.children[1], columns, indentation + '        ')
        else:
            for value in node.children:
                print(f'{indentation}[{column.column_name} = {value}]', file=output)
                __print_tree(output, node.children[value], columns, indentation + '        ')
                

def print_tree(tree, file=stdout):
    __print_tree(file, tree.root, tree.columns, '')
