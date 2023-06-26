import requests
from hathitrust_api import DataAPI
from hathitrust_api import BibAPI
from PIL import Image, ImageTk
import io
import cv2
import string
import tkinter
from tkinter import simpledialog
from scipy import stats
from requests_oauthlib import OAuth1
import pytesseract
import input_page_nums
import statistics


def remove_punct(s):
    return str(s).translate(str.maketrans('', '', string.punctuation))

def getImage(p, path):
    url = "".join(["https://babel.hathitrust.org/cgi/htd/volume/pageimage/", doc_id, "/", str(p)])
    r = rsession.get(url, params={'v': str(2), 'height': str(3400)})
    r.raise_for_status()
    im = r.content
    im = Image.open(io.BytesIO(im))
    im.save(path)
    return im

access_key = open("access_key.txt").read().replace("\n", "")
secret_key = open("secret_key.txt").read().replace("\n", "")
doc_id = 'mdp.39015006977725'
data_api = DataAPI(access_key, secret_key)
bib_api = BibAPI()
bib_info = bib_api.get_single_record_json('htid', doc_id, full = True)
title = str([bib_info["records"][r]["titles"] for r in bib_info["records"]][0][0])
date = str([bib_info["records"][r]["publishDates"] for r in bib_info["records"]][0][0])
dates = []
#

oauth = OAuth1(client_key=access_key,
                            client_secret=secret_key,
                            signature_type='query')

rsession = requests.Session()
rsession.auth = oauth



#loop through ocr pages until page contains "contents"
p_num = 0
ocr = ""
while p_num < 1000 and "contents" not in ocr.lower():
    p_num += 1
    ocr = str(data_api.getpageocr(doc_id, p_num))
    tokens = remove_punct(ocr).split(" ")
    [dates.append(token) for token in tokens if len(token) == 4 and token.isnumeric() and token[0]=='1' and int(token[1]) > 6]


#then pull page numbers for the section we want, imports of merchandise by articles and countries
date = statistics.mode(dates)
toc_im = getImage(p_num, "".join(["/home/emily/Downloads/", date, "/toc.tiff"]))
current_page = [int(i) for i in ocr if i.isdigit()][0]
page_offset = p_num - current_page
section_starts = -1
section_ends = -1
l = 0

lines = ocr.split("\\n")
while l < len(lines) and section_starts == -1:
    line = lines[l]

    if "Imports of merchandise, by articles and countries" in line:
        try:
            nums = [token for token in str(line).translate(str.maketrans('', '', string.punctuation)).split(" ") if token.isnumeric()]
            #nums = [token for token in line.replace("-", " ").split(" ") if token.isnumeric()]
            section_starts = int(nums[-1]) + page_offset
            section_ends = int(str(lines[l+1]).translate(str.maketrans('', '', string.punctuation)).split(" ")[-1]) + page_offset
        except IndexError:
            print("Error: couldn't find page numbers")

        section_starts, section_ends = [int(num) + page_offset for num in input_page_nums.main(max(section_starts-page_offset, 0), max(section_ends - page_offset, 0),
                                               "".join(["/home/emily/Downloads/", date, "/toc.tiff"]))]
    else:
        l += 1

#download images for those pages
for p in range(section_starts, section_ends):
    print("downloading page " + str(p))
    getImage(p, "".join(["/home/emily/Downloads/", date, "/temp_images/", str(p), ".tiff"]))

print("complete")