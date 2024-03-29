import argparse
import pandas as pd
import math
import os

def clean_str(s):
    if ((isinstance(s, int) or isinstance(s, float)) and not math.isnan(s)) or (isinstance(s, str) and s.isnumeric()):
        return str(int(s)).replace("\n", "").replace("$", "").replace(",", "").replace(" ", "").replace(".", "")
    else:
        return str(s).replace("\n", "").replace("$", "").replace(",", "").replace(" ", "").replace(".", "")

def set_value(i, c, val, conf):
    if "doll" in str(columns[c]).lower():
        output[i]["value"] = val
        output[i]["conf_value"] = conf
        try:
            output[i]["value_measure"] = str(columns[c]).split("\n\n")[-2]
        except IndexError:
            output[i]["value_measure"] = ""
    else:
        output[i]["quantity"] = val
        output[i]["conf_quantity"] = conf
        try:
            output[i]["quantity_measure"] = str(columns[c]).split("\n\n")[-2]
        except IndexError:
            output[i]["quantity_measure"] = ""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
    help="path to directory to be parsed")
args = vars(ap.parse_args())

files = sorted(os.listdir(args["directory"]))

output_path = ""

for f in files:
    file_path = "/".join([args["directory"], f])
    csv = pd.read_csv(file_path)
    table = csv.values.tolist()
    columns = csv.columns.tolist()

    #find cells that are likely to contain relevant data
    conf_cols = [c for c in range(0, len(table[0])) if columns[c].replace("col", "").replace("CONF", "").isnumeric()]
    contain_data = [[clean_str(row[c]).isnumeric() and c not in conf_cols for c in range(0, len(row))] for row in table]
    contain_data_row = [any(row) for row in contain_data]
    contain_data_col = [any([row[c] for row in contain_data]) for c in range(0, len(contain_data[0]))]
    confidences = [[row[c] for row in table] for c in conf_cols]


    #table originally organized as row representing country/locale and column representing imported merchandise
    #want to translate into output that is organized as individual import instances (i.e furs from Argentina, gold from Germany, etc)
    #loop through rows and add to output when a given column has a value

    output = []
    for row in table:
        for c in range(0, len(row)):
            if contain_data_col[c] and (type(row[c])==str or not math.isnan(row[c])) and not "total" in str(row[0]).lower():

                #check if this has already been added, in that case just alter appropriate columns
                duplicates = []
                if len(output) > 0:
                    duplicates = [r["description"]=="\n\n".join(columns[c].split("\n\n")[0:-2]) and r["origin"] == row[0] for r in output]
                if len(output) > 0 and len(duplicates) > 0 and any(duplicates):
                    set_value(duplicates.index(True), c, row[c], row[c+1])
                else:
                    output.append(dict(origin = row[0], conf_origin = row[1], destination = "United States of America",
                                       value = None, conf_value = None, value_measure = None,
                                       quantity = None, conf_quantity = None, quantity_measure = None,
                                       description = "\n\n".join(columns[c].split("\n\n")[0:-2]),
                                       report_name = None,
                                       year = file_path.split("/")[-3],
                                       page_number = f.replace(".csv", "")))
                    set_value(-1, c, row[c], row[c+1])


    df = pd.DataFrame(data = output)

    if f == files[0]:
        output_path = "".join([args["directory"].replace(args["directory"].split("/")[-1], "").replace("//", "/"), "output.csv"])
        df.to_csv(output_path, index=False, header = list(output[0].keys()))
    else:
        df.to_csv(output_path, mode='a', index=False, header = False)