import argparse
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
    help="path to input csv to be parsed")
args = vars(ap.parse_args())

df = pd.read_csv(args["file"])

#columns: source, destination, description, quantity, quantity measure,
#           value, weight, weight measure, item category, year, report name,
#           page number


