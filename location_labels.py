import pandas as pd
import argparse
import os
import numpy
import string
import datetime
from jiwer import cer, wer
from SPARQLWrapper import SPARQLWrapper, JSON


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
    help="path to directory containing location labels")
ap.add_argument("-i", "--input", required=True,
    help="path to file containing labels to clean")
args = vars(ap.parse_args())

stop_words = ["Kingdom of", "Emirate of"]

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
        return results_list
    except Exception as e:
        return e


def get_territories():
    #query "integral overseas territories"
    return query("""
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

            SELECT DISTINCT ?titleLabel ?subtitleLabel ?loc ?loc2 WHERE {
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
              {

                SELECT DISTINCT ?title ?subtitle ?loc ?loc2 WHERE {

                ?title p:P31|p:P279 ?isTerritory.
                ?isTerritory ps:P31|ps:P279 wd:Q26934845.

                OPTIONAL {
                  ?subtitle p:P31 ?hasSubtitle.
                  ?hasSubtitle ps:P31 ?title.
                  
                  OPTIONAL {
                    ?subtitle p:P625 ?hasLoc.
                    ?hasLoc ps:P625 ?loc
                  }
               }
               
               OPTIONAL{
                  ?title p:P625 ?hasLoc.
                  ?hasLoc ps:P625 ?loc.
                }
              }
            }
            }""")


def get_canadian_provinces():
    # query canadian provinces"
    return query("""
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

            SELECT DISTINCT ?provinceLabel ?loc WHERE {
              SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
              {

                SELECT DISTINCT ?province ?loc WHERE {

                ?province p:P31|p:P279 ?isProvince.
                ?isProvince ps:P31|ps:P279 wd:Q11828004.

               OPTIONAL{
                  ?province p:P625 ?hasLoc.
                  ?hasLoc ps:P625 ?loc.
                }
              }
            }
            }""")

def get_country_subdivisions(date):
    return query(f"""
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

            SELECT DISTINCT ?regionLabel ?region_beginning ?region_end ?cLabel ?loc WHERE {{
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
              {{
            
                SELECT DISTINCT ?region ?region_beginning ?region_end ?c ?loc WHERE {{
               
                  ?title p:P31|p:P279 ?isCountrySubdivision.
                  ?isCountrySubdivision ps:P31|ps:P279 wd:Q10864048.
                  
                  ?title p:P17 ?whichCountry.
                  ?whichCountry ps:P17 ?c.
                  
                  ?region p:P31 ?instanceOfCountrySubdivision.
                  ?instanceOfCountrySubdivision (ps:P31) ?title.
                  
                  OPTIONAL {{
                    ?region p:P625 ?hasLoc.
                    ?hasLoc ps:P625 ?loc.
                  }}
                  
                  OPTIONAL {{
                    ?c p:P625 ?hasLoc.
                    ?hasLoc ps:P625 ?loc.
                   }}
                  
                  FILTER(?c IN (?country)) {{
                    SELECT DISTINCT ?country WHERE {{
                    ?country p:P31 ?isSovereign.
                    ?isSovereign ps:P31 wd:Q3624078.
                   
                    OPTIONAL{{
                      ?country p:P571 ?startCorrectTime.
                      ?startCorrectTime psv:P571 ?startCorrectTimeValue.
                      ?startCorrectTimeValue wikibase:timeValue ?beginning.
                    }}
                  
                    OPTIONAL {{
                      ?country p:P576 ?endCorrectTime.
                      ?endCorrectTime psv:P576 ?endCorrectTimeValue.
                      ?endCorrectTimeValue wikibase:timeValue ?end.
                    }}
                  
                    FILTER(?beginning < "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime &&
                       (?end > "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime || !BOUND(?end)))
                }}}}
                  
                OPTIONAL{{
                      ?region p:P571 ?regionStartCorrectTime.
                      ?regionStartCorrectTime psv:P571 ?regionStartCorrectTimeValue.
                      ?regionStartCorrectTimeValue wikibase:timeValue ?region_beginning.
                }}
                  
                OPTIONAL{{
                      ?region p:P576 ?regionEndCorrectTime.
                      ?regionEndCorrectTime psv:P576 ?regionEndCorrectTimeValue.
                      ?regionEndCorrectTimeValue wikibase:timeValue ?region_end.
                }}
                #FILTER(?region_beginning < "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime &&
                       #(?region_end > "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime || !BOUND(?region_end)))
              }}
            }}
            }}""")


def get_countries(date):
    return query(f"""SELECT DISTINCT ?countryLabel ?loc WHERE {{
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      {{
            SELECT DISTINCT ?country ?loc WHERE {{
            ?country p:P31 ?statement3.
            ?statement3 ps:P31 wd:Q3624078.
           
            OPTIONAL{{
              ?country p:P571 ?statement_4.
              ?statement_4 psv:P571 ?statementValue_4.
              ?statementValue_4 wikibase:timeValue ?beginning.
            }}
          
            OPTIONAL {{
              ?country p:P576 ?statement_5.
              ?statement_5 psv:P576 ?statementValue_5.
              ?statementValue_5 wikibase:timeValue ?end2.
            }}
            
            OPTIONAL {{
                ?country p:P625 ?hasLoc.
                ?hasLoc ps:P625 ?loc.
              }}
          
            FILTER(?beginning < "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime &&
              (?end2 > "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime || !BOUND(?end2)))
        }}}}
    }}""")


def get_geographic_regions(date):
    return query(f"""
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

            SELECT DISTINCT ?regionLabel ?region_beginning ?region_end ?loc WHERE {{
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
              {{
            
                SELECT DISTINCT ?region ?region_beginning ?region_end ?loc WHERE {{
                
                  ?region p:P31|p:P279 ?isRegion.
                  ?isRegion ps:P31|ps:P279 wd:Q82794.
                  
                  OPTIONAL {{
                    ?region p:P625 ?hasLoc.
                    ?hasLoc ps:P625 ?loc.
                  }}
                  
                  OPTIONAL{{
                      ?region p:P571 ?startCorrectTime.
                      ?startCorrectTime psv:P571 ?startCorrectTimeValue.
                      ?startCorrectTimeValue wikibase:timeValue ?region_beginning.
                  }}
                  
                  OPTIONAL{{
                      ?region p:P576 ?endCorrectTime.
                      ?endCorrectTime psv:P576 ?endCorrectTimeValue.
                      ?endCorrectTimeValue wikibase:timeValue ?region_end.
                  }} 
                  #FILTER(?region_beginning < "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime &&
                       #(?region_end > "+{str(date).split(" ")[0]}T{str(date).split(" ")[1]}Z"^^xsd:dateTime || !BOUND(?region_end)))
            
              }}
            }}
            }}""")
date = datetime.datetime.strptime(str(1900), "%Y")
territories = get_territories()
territories = [row for row in territories if "Point(" in str(row)]
ca_provinces = get_canadian_provinces()
[territories.append(row + ["nan"]) for row in ca_provinces]
countries = get_countries(date)
countries = [row for row in countries if "Point(" in str(row)]

subdivisions = get_country_subdivisions(date)
subdivisions = [row for row in subdivisions if "Point(" in str(row) and not str(row[3]).replace("Q", "").isnumeric()]

geo_words = get_geographic_regions(date)
geo_words = [row for row in geo_words if "Point(" in str(row)]

df = pd.DataFrame(data = territories)
df.to_csv("".join([str(args["directory"]), "/territories.csv"]), index = False, header = False)

df = pd.DataFrame(data = countries)
df.to_csv("".join([str(args["directory"]), "/countries.csv"]), index = False, header = False)

df = pd.DataFrame(data = subdivisions)
df.to_csv("".join([str(args["directory"]), "/subdivisions.csv"]), index = False, header = False)

df = pd.DataFrame(data = geo_words)
df.to_csv("".join([str(args["directory"]), "/geo_words.csv"]), index = False, header = False)

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


c_names = [row[0] for row in countries]
s_names = [row[3] for row in subdivisions]
t_names = [row[1] if str(row[2]) == "nan" else row[2] for row in territories]

c_vocab = []
[[c_vocab.append(word) for word in loc.split(" ")] for loc in c_names]
c_vocab = unique(c_vocab)
c_vocab = [word for word in c_vocab if any(c.isalpha() for c in word)]

s_vocab = []
[[s_vocab.append(word) for word in loc.split(" ")] for loc in s_names]
s_vocab = unique(s_vocab)
s_vocab = [word for word in s_vocab if any(c.isalpha() for c in word)]

t_vocab = []
[[t_vocab.append(word) for word in loc.split(" ")] for loc in t_names]
t_vocab = unique(t_vocab)
t_vocab = [word for word in t_vocab if any(c.isalpha() for c in word)]

input = pd.read_csv(args["input"])
input_values = input.values.tolist()
input_columns = input.columns.tolist()
input_countries = [row[input_columns.index("origin")] for row in input_values]
input_countries = [c.replace("\n", "") for c in input_countries]
input_countries = [c for c in input_countries if c != ""]

label_cers = []
label_coords = []
coords_cer = []

c_ind = 0
for c_ind in range(0, len(input_countries)):
    c = input_countries[c_ind]
    c_by_word = [word for word in c.split(" ") if word]

    w_ind = 0
    word_cers = []
    for w_ind in range(0, len(c_by_word)):
        word = c_by_word[w_ind]

        c_cers = [cer(word, loc) for loc in c_vocab]
        low_c_cer = min(c_cers)
        c_match = c_vocab[c_cers.index(low_c_cer)]

        s_cers = [cer(word, loc) for loc in s_vocab]
        low_s_cer = min(s_cers)
        s_match = s_vocab[s_cers.index(low_s_cer)]

        t_cers = [cer(word, loc) for loc in t_vocab]
        low_t_cer = min(t_cers)
        t_match = t_vocab[t_cers.index(low_t_cer)]

        low_cer = min(low_c_cer, low_s_cer, low_t_cer)
        word_cers.append(low_cer)

        if low_cer < 0.3:
            if low_cer == low_c_cer:
                c_by_word[w_ind] = c_match
            else:
                if low_cer == low_s_cer:
                    c_by_word[w_ind] = s_match
                else:
                    if low_cer == low_t_cer:
                        c_by_word[w_ind] = t_match

        #print("".join([word, " MATCHES WITH \n\t",
                      #c_match, " CER ", str(low_c_cer), "\n\t",
                       #s_match, " CER ", str(low_s_cer), "\n\t",
                       #t_match, " CER ", str(low_t_cer)]))
    label_cers.append(word_cers)
    input_countries[c_ind] = " ".join(c_by_word)

    full_c_cers = [cer(input_countries[c_ind], loc) for loc in c_names]
    low_full_c_cer = min(full_c_cers)

    full_s_cers = [cer(input_countries[c_ind], loc) for loc in s_names]
    low_full_s_cer = min(full_s_cers)

    full_t_cers = [cer(input_countries[c_ind], loc) for loc in t_names]
    low_full_t_cer = min(full_t_cers)

    low_full_cer = min(low_full_c_cer, low_full_s_cer, low_full_t_cer)

    coords_cer.append(low_full_cer)

    countries_exact_match = [input_countries[c_ind] in c for c in c_names]
    if any(countries_exact_match):
        label_coords.append(countries[[m for m in range(0, len(countries_exact_match)) if countries_exact_match[m]][0]][1])
    else:
        subdivisions_exact_match = [input_countries[c_ind] in s_names]
        if any(subdivisions_exact_match):
            label_coords.append(subdivisions[[m for m in subdivisions_exact_match if m][0]][2])
        else:
            territories_exact_match = [input_countries[c_ind] in t_names]
            if any(territories_exact_match):
                label_coords.append(territories[[m for m in territories_exact_match if m][0]][0])
            else:
                if low_full_cer < 2.0:
                    if low_full_cer == low_full_c_cer:
                        label_coords.append(countries[full_c_cers.index(low_full_c_cer)][1])
                    else:
                        if low_full_cer == low_full_s_cer:
                            label_coords.append(subdivisions[full_s_cers.index(low_full_s_cer)][2])
                        else:
                            if low_full_cer == low_full_t_cer:
                                label_coords.append(territories[full_t_cers.index(low_full_t_cer)][0])
                else:
                    label_coords.append("")


    #print(c_wers)

# #load files
# files = os.listdir(args["directory"])
#
# output_directory = args["directory"].replace(args["directory"].split("/")[-1], "images_as_csv")
#
# input = pd.read_csv(args["input"])
# input_values = input.values.tolist()
# input_columns = input.columns.tolist()
# input_countries = [row[input_columns.index("origin")] for row in input_values]
#
# locations_list = []
#
# for file in files:
#     file_path = "/".join([args["directory"], file])
#     csv = pd.read_csv(file_path)
#     table = csv.values.tolist()
#     columns = csv.columns.tolist()
#
#     lab_in = columns.index("label")
#
#     if "sublabel" in columns:
#         sub_in = columns.index("sublabel")
#         [locations_list.append(str(row[lab_in])) if str(row[sub_in]) == "nan" else locations_list.append(str(row[sub_in])) for row in table]
#     else:
#         [locations_list.append(str(row[lab_in])) for row in table]
#
# locations_list = [loc for loc in locations_list if not loc.replace('Q', "").isnumeric()]
# locations_vocab = []
# [[locations_vocab.append(word) for word in loc.split(" ")] for loc in locations_list]
# locations_vocab = unique(locations_vocab)
# input_countries = [c.replace("\n", "") for c in input_countries]
# # locations_list = [remove_stop_words(loc) for loc in locations_list]
# # input_countries = [remove_stop_words(c) for c in input_countries]
#
# c_ind = 0
# for c_ind in range(0, len(input_countries)):
#     c = input_countries[c_ind]
#     c_by_word = []
#     [c_by_word.append(word) for word in c.split(" ") if word]
#
#     w_ind = 0
#     for w_ind in range(0, len(c_by_word)):
#         word = c_by_word[w_ind]
#         cers = [cer(word, loc) for loc in locations_vocab]
#         low_cer = min(cers)
#         match = locations_vocab[cers.index(low_cer)]
#         #print("".join([word, " MATCHES WITH ", match, " CER ", str(low_cer)]))
#         c_by_word[w_ind] = match
#     input_countries[c_ind] = " ".join(c_by_word)

output_values = [[input_countries[r]] + [label_coords[r]] + [coords_cer[r]] + input_values[r] for r in range (0, len(input_values))]
output_columns = ["clean_origin"] + ["coordinates"] + ["coordinate certainty"] + input_columns

df = pd.DataFrame(data = output_values)
df.to_csv(str(args["input"]).replace("outputsmall.csv", "output_clean.csv"), index=False, header = output_columns)


