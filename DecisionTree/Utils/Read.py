from csv import reader

def read_csv(file):
    with open(file) as csv_file:
        csv_reader = reader(csv_file, delimiter = ',')

        data = []
        for row_index, row in enumerate(csv_reader):
            if row_index == 0:
                columns = row
            else:
                for column_index, column_name in enumerate(columns):
                    if column_name.startswith('n') or column_name.startswith('N'):
                        row[column_index] = float(row[column_index])

                data.append(row)
                
        return columns, data
        