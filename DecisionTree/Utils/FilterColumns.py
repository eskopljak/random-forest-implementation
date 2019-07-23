def filter_columns(columns, data, column_indices):
    new_columns = []
    for i in column_indices:
        new_columns.append(columns[i])

    new_data = []
    for row in data:
        new_data.append([row[i] for i in column_indices])
    
    return new_columns, new_data


def set_target(columns, data, target_column):
    temp = columns[target_column]
    columns[target_column] = columns[-1]
    columns[-1] = temp

    for row in data:
        temp = row[target_column]
        row[target_column] = row[-1]
        row[-1] = temp