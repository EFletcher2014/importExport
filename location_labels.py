import pandas as pd
import argparse
import string
import datetime
from jiwer import cer
from SPARQLWrapper import SPARQLWrapper, JSON

query_string = """
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wds: <http://www.wikidata.org/entity/statement/>
            PREFIX wdv: <http://www.wikidata.org/value/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX bd: <http://www.bigdata.com/rdf#>

            PREFIX wdref: <http://www.wikidata.org/reference/>
            PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
            PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
            PREFIX pqv: <http://www.wikidata.org/prop/qualifier/value/>
            PREFIX pqn: <http://www.wikidata.org/prop/qualifier/value-normalized/>
            PREFIX pr: <http://www.wikidata.org/prop/reference/>
            PREFIX prv: <http://www.wikidata.org/prop/reference/value/>
            PREFIX prn: <http://www.wikidata.org/prop/reference/value-normalized/>
            PREFIX wdno: <http://www.wikidata.org/prop/novalue/>
            PREFIX wdata: <http://www.wikidata.org/wiki/Special:EntityData/>

            PREFIX schema: <http://schema.org/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX prov: <http://www.w3.org/ns/prov#>
            PREFIX bds: <http://www.bigdata.com/rdf/search#>
            PREFIX gas: <http://www.bigdata.com/rdf/gas#>
            PREFIX hint: <http://www.bigdata.com/queryHints#>

            SELECT DISTINCT ?itemLabel ?superClassLabel ?loc WHERE {
                FILTER(!CONTAINS(LCASE(?itemLabel), " tribe") && !CONTAINS(LCASE(?itemLabel), " indian"))
                    {SELECT DISTINCT ?itemLabel ?superClassLabel ?loc WHERE {
                        SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
                        {
                            SELECT DISTINCT ?item ?superClass ?loc WHERE {
                            {
                                ?item (p:P31/p:P279*/ps:P31/ps:P279*)|(wdt:P31/wdt:P279*) ?superClass.
                  
                                ?item p:P625 ?hasLoc.
                                ?hasLoc ps:P625 ?loc.
                                      
                                VALUES ?superClass {wd:Q6256}
                              }
                              UNION
                              {
                                ?item (p:P31/p:P279*/ps:P31/ps:P279*)|(wdt:P31/wdt:P279*) ?superClass.
                                
                                ?item p:P625 ?hasLoc.
                                ?hasLoc ps:P625 ?loc.
                                      
                                VALUES ?superClass {wd:Q7275}
                              }
                              UNION
                              {
                                ?item (p:P31/p:P279*/ps:P31/ps:P279*)|(wdt:P31/wdt:P279*) ?superClass.
                                
                                ?item p:P625 ?hasLoc.
                                ?hasLoc ps:P625 ?loc.
                                      
                                VALUES ?superClass {wd:Q26934845}
                              }
                              UNION
                              {
                                ?item (p:P31/p:P279*/ps:P31/ps:P279*)|(wdt:P31/wdt:P279*) ?superClass.
                                
                                ?item p:P625 ?hasLoc.
                                ?hasLoc ps:P625 ?loc.
                                      
                                VALUES ?superClass {wd:Q107390}
                              }
                            }
                        }
                    }
                }
            }"""


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
    help="path to directory containing location labels")
ap.add_argument("-i", "--input", required=True,
    help="path to file containing labels to clean")
ap.add_argument("-c", "--correctLabels", required=True,
                help="T if country names should be corrected, F if they should be maintained")
args = vars(ap.parse_args())

stop_words = ["new", "imperial", "emirate", "french", "british", "dutch", "danish", "swedish", "the", "other", "colonies", "united", "kingdom", "of", "empire", "north", "south", "east", "west", "and"]

def query(q_string):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # gets the first 3 geological ages
    # from a Geological Timescale database,
    # via a SPARQL endpoint
    sparql.setQuery(q_string)

    try:
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        results_df = pd.json_normalize(results['results']['bindings'])
        contains_data_results = ["value" in col for col in results_df.columns]
        results_list = [[row[c] for c in range(0, len(row)) if contains_data_results[c]] for row in results_df.values]
        return results_list, [col.replace(".value", "") for col in results_df.columns if "value" in col]
    except Exception as e:
        return e

def remove_stop_words(s):
    for word in stop_words:
        s = s.replace(word, "")
    return s

def unique(list1):
    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def clean_labels(list, comp):
    c_ind = 0
    label_cers = []
    clean_list = []
    for c_ind in range(0, len(list)):
        c = list[c_ind]
        c_by_word = [word for word in c.split(" ") if word]

        merge_in = 0
        while merge_in < len(c_by_word)-1:
            if len(c_by_word[merge_in]) == 1 or len(c_by_word[merge_in+1]) == 1:
                c_by_word[merge_in] = "".join([c_by_word[merge_in], c_by_word[merge_in + 1]])
                c_by_word.remove(c_by_word[merge_in + 1])
            else:
                merge_in += 1
        #c_by_word = [["".join([row[w], row[w+1]]) for w in range(0, len(row)-1) if len(row[w]) == 1 or len(row[w+1]) == 1] for row in c_by_word]

        w_ind = 0
        word_cers = []
        for w_ind in range(0, len(c_by_word)):
            word = c_by_word[w_ind]

            c_cers = [cer(word, loc) for loc in comp]
            low_cer = min(c_cers)
            c_match = comp[c_cers.index(low_cer)]

            word_cers.append(low_cer)

            if low_cer < 0.33:
                c_by_word[w_ind] = c_match

            # print("".join([word, " MATCHES WITH \n\t",
            # c_match, " CER ", str(low_c_cer), "\n\t",
            # s_match, " CER ", str(low_s_cer), "\n\t",
            # t_match, " CER ", str(low_t_cer)])
        label_cers.append(word_cers)
        clean_list.append(" ".join(c_by_word))

    return clean_list, label_cers

def get_coords(list, compare_values):
    num = 0
    cers = []
    coords = []
    for label in list:
        label = label.strip()
        if label in compare_values:
            num += 1
            coords.append(compare[compare_values.index(label)][c_cols.index("loc")])
            cers.append(0)
        else:
            partial_matches, partial_matches_ind, contained = get_partial_matches(label, compare_values)
            if partial_matches:
                if any(contained):
                    num += 1
                    closest = partial_matches_ind[contained.index(True)]
                    coords.append(compare[closest][c_cols.index("loc")])
                    cers.append(1 - (len(compare[closest][c_cols.index("itemLabel")].split(" "))/len(label.split(" "))))
                else:
                    num += 1
                    coords.append(compare[partial_matches_ind[0]][c_cols.index("loc")])
                    cers.append(cer(compare[partial_matches_ind[0]][c_cols.index("itemLabel")], label))
            else:
                coords.append("Point(0 0)")
                cers.append(1)
    return coords, cers, num/len(list)

def get_partial_matches(label, compare):
    partial_matches_ind = [v for v in range(0, len(compare)) if
                       any([l.lower() in compare[v].lower() for l in label.split(" ") if
                            l and l.lower() not in stop_words and len(l) > 2])]
    partial_matches = [compare[ind] for ind in partial_matches_ind]

    in_label = [match in label for match in partial_matches]

    return partial_matches, partial_matches_ind, in_label


def map_values(lt, ln, old_labels, new_labels, cers, dest, d_lt, d_ln, orig_values, orig_cols):
    output_values = []
    for r in range(0, len(orig_values)):
        output_values.append([new_labels[old_labels.index(orig_values[r][orig_cols.index("origin")].replace("\n", ""))]] +
                              [lt[old_labels.index(orig_values[r][orig_cols.index("origin")].replace("\n", ""))]] +
                              [ln[old_labels.index(orig_values[r][orig_cols.index("origin")].replace("\n", ""))]] +
                              [cers[old_labels.index(orig_values[r][orig_cols.index("origin")].replace("\n", ""))]] +
                              [d_lt[dest.index(orig_values[r][orig_cols.index("destination")].replace("\n", ""))]] +
                              [d_ln[dest.index(orig_values[r][orig_cols.index("destination")].replace("\n", ""))]] +
                              orig_values[r])

    return output_values

input = pd.read_csv(args["input"])
input_values = input.values.tolist()
input_columns = input.columns.tolist()
input_columns = [str(col).lower() for col in input_columns]
input_values = [[row[c] if input_columns[c] != "origin" else str(row[c]).translate(str.maketrans(' ', ' ', string.punctuation)) for c in range(0, len(row))] for row in input_values]

input_years = [row[input_columns.index("year")] for row in input_values]
unique_years = unique(input_years)
unique_years = [datetime.datetime.strptime(str(year), "%Y") for year in unique_years]
input_countries = [row[input_columns.index("origin")] for row in input_values]
input_countries = [c.replace("\n", "") for c in input_countries]
input_countries = [c for c in input_countries if c != ""]
input_countries = unique(input_countries)

dest_countries = [row[input_columns.index("destination")] for row in input_values]
dest_countries = [str(c).replace("\n", "") for c in dest_countries]
dest_countries = [c for c in dest_countries if c != ""]
dest_countries = unique(dest_countries)

correct_labels = args["correctLabels"]

if correct_labels == "T":
    correct_labels = True
else:
    correct_labels = False

label_coords = []
coords_cer = []

compare, c_cols = query(query_string)


compare = [row for row in compare if "Point(" in str(row)]

df = pd.DataFrame(data = compare)
df.to_csv("".join([str(args["directory"]), "/compare.csv"]), index = False, header = False)

c_names = [row[c_cols.index("itemLabel")] for row in compare]
c_vocab = []
[[c_vocab.append(word) for word in loc.split(" ")] for loc in c_names]
c_vocab = unique(c_vocab)
c_vocab = [word for word in c_vocab if any(c.isalpha() for c in word)]

#first, fix labels
label_cers = [0 for c in input_countries]

if correct_labels:
    new_labels, label_cers = clean_labels(input_countries, c_vocab)

    print(input_countries)
else:
    new_labels = input_countries


# #then, find coordinates
label_coords, coord_cer, perc = get_coords(new_labels, c_names)

dest_coords, dest_coord_cer, dest_perc = get_coords(dest_countries, c_names)

if correct_labels:
    coord_cer = [coord_cer[i] + sum(label_cers[i]) for i in range(0, len(label_cers))]

coord_cer = [max([0, 1 - c]) for c in coord_cer]

print(label_coords)

label_coords = [coord.replace("Point", "").replace("(", "").replace(")", "").split(" ") for coord in label_coords]
long = [coord[0] for coord in label_coords]
lat = [coord[1] for coord in label_coords]

dest_coords = [coord.replace("Point", "").replace("(", "").replace(")", "").split(" ") for coord in dest_coords]
d_long = [coord[0] for coord in dest_coords]
d_lat = [coord[1] for coord in dest_coords]

output_values = map_values(lat, long, input_countries, new_labels, coord_cer, dest_countries, d_lat, d_long, input_values, input_columns)

output_columns = ["clean_origin"] + ["latitude"] + ["longitude"] + ["coordinate certainty"] + ["destination_latitude"] + ["destination_longitude"] + input_columns

df = pd.DataFrame(data = output_values)
df.to_csv(str(args["input"]).replace(".csv", "_clean.csv"), index=False, header = output_columns)


