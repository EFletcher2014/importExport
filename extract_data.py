import argparse
import pandas as pd

def clean_str(s):
    return str(s).replace("\n", "").replace("$", "").replace(",", "").replace(" ", "").replace(".", "")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
    help="path to input csv to be parsed")
args = vars(ap.parse_args())

table = pd.read_csv(args["file"]).values.tolist()

#find cells that are likely to contain relevant data
contain_data = [[clean_str(cell).isnumeric() for cell in row] for row in table]
contain_data_row = [any(row) for row in contain_data]
contain_data_col = [any([row[c] for row in contain_data]) for c in range(0, len(contain_data[0]))]
data = [table[r] for r in range(0, len(table)) if contain_data_row[r] == True]
#[[table[r][c] for c in range(0, len(table[0])) if contain_data_col[c] == True] for r in range(0, len(table)) if contain_data_row[r] == True]

clean_data = [[clean_str(data[r][c]) if contain_data_col[c] == True else data[r][c] for c in range(0, len(data[0]))] for r in range(0, len(data))]



#columns: source, destination, description, quantity, quantity measure,
#           value, weight, weight measure, item category, year, report name,
#           page number

output = dict(source_historic_name = "", )
