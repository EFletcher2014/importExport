from hathitrust_api import DataAPI
from hathitrust_api import BibAPI
from PIL import Image
import io
import cv2
from scipy import stats
import pytesseract

def clean_cell(x, y, w, h):
    cell_im = img_bin_otsu[y:y+h, x:x+w]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    cell_im = cv2.erode(cell_im, kernel, iterations=1)
    cell_im = cv2.dilate(cell_im, kernel, iterations=1)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(cell_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # check if contour is at least 35px wide and 100px tall, and if
        # so, consider the contour a digit
        if h < 2 or w < 2:
            cell_im = cv2.drawContours(cell_im, [c], 0, (0, 0, 0), -1)

    # cv2.imshow("clean", cell_im)
    # cv2.waitKey(0)

    return cell_im

access_key = open("access_key.txt").read().replace("\n", "")
secret_key = open("secret_key.txt").read().replace("\n", "")
doc_id = 'mdp.39015006977725'
data_api = DataAPI(access_key, secret_key)
bib_api = BibAPI()
bib_info = bib_api.get_single_record_json('htid', 'mdp.39015006977725', full = True)
title = str([bib_info["records"][r]["titles"] for r in bib_info["records"]][0][0])
date = str([bib_info["records"][r]["publishDates"] for r in bib_info["records"]][0][0])


#loop through ocr pages until page contains "contents"
p_num = 0
ocr = ""
while p_num < 1000 and "contents" not in ocr.lower():
    p_num += 1
    ocr = str(data_api.getpageocr('mdp.39015006977725', p_num))

#then pull page numbers for the section we want, imports of merchandise by articles and countries
current_page = [int(i) for i in ocr if i.isdigit()][0]
page_offset = p_num - current_page
section_starts = -1
section_ends = -1
l = 0

lines = ocr.split("\\n")
while l < len(lines) and section_starts == -1:
    line = lines[l]
    if "Imports of merchandise, by articles and countries" in line:
        section_starts = int(line.replace("-", " ").split(" ")[-1]) + page_offset
        section_ends = int(lines[l+1].replace("-", " ").split(" ")[-1])-1 + page_offset
    else:
        l += 1

#download images for those pages
for p in range(section_starts, section_ends):
    im = data_api.getpageimage(doc_id, p)
    im = Image.open(io.BytesIO(im))
    im = im.save("".join([doc_id, "page", p, ".tiff"]))

# #then pull page numbers for section we want--imports of merchandise by articles and countries
# con_im = data_api.getpageimage(doc_id, p_num)
# con_im = Image.open(io.BytesIO(con_im))
# con_im = con_im.save("".join([doc_id, ".tiff"]))
#
# image = cv2.imread("".join([doc_id, ".tiff"]))
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_bin_otsu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
# cv2.imshow('original', image)
# cv2.imshow('thresh', img_bin_otsu)
# cv2.waitKey(0)
#
# # Dilate to combine adjacent text contours
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# dilate = cv2.dilate(img_bin_otsu, kernel, iterations=2)
#
# cv2.imshow("dilate", dilate)
# cv2.waitKey(0)
#
# # Find contours, highlight text areas, and extract ROIs
# cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# #get all possible lines
# text_lines = []
# hierarchies = []
#
# for i in range(0, len(cnts)):
#     c = cnts[i]
#     area = cv2.contourArea(c)
#     x, y, w, h = cv2.boundingRect(c)
#     hier = hierarchy[0][i]
#
#     if area > 300 and w > h and h > image.shape[0] / 90:
#         # cv2.imshow("text line", cv2.resize(
#         #     image[y + columns[0]["start_y"] + columns[0]["height"]:y + columns[0]["star
#         #     t_y"] + columns[0]["height"] + h,
#         #     table_x:table_x1], None, fx=0.25, fy=0.25))
#         # cv2.waitKey(0)
#         l_start = y
#         l_end = y + h
#
#         test = []
#
#
#         text_lines.insert(0, [l_start, l_end])
#         hierarchies.insert(0, hier)
#
# line_im = image.copy()
# for line in text_lines:
#     line_im = cv2.rectangle(line_im, (0, line[0]), (1000, line[1]), color = (255, 255, 0), thickness = 1)
#
#
# cv2.imshow("lillnes", line_im)
# cv2.waitKey(0)

# #calculate suspected line height using mode, remove lines that are too short
# line_height = stats.mode([line[1] - line[0] for line in text_lines])[0]
#
# text_lines_no_short = [line for line in text_lines if line[1] - line[0] >= line_height - 2]
#
# #remove duplicates
# text_lines_no_dup = []
# for line in text_lines_no_short:
#     if line not in text_lines_no_dup:
#         text_lines_no_dup.append(line)
#
# line_im = image.copy()
# for line in text_lines_no_dup:
#     line_im = cv2.rectangle(line_im, (0, line[0]), (1000, line[1]), color = (255, 255, 0), thickness = 1)
#     cv2.imshow("row", clean_cell(0, line[0], img_bin_otsu.shape[1], line[1]-line[0]))
#     cv2.waitKey(0)
#
#     label = str(pytesseract.image_to_string(clean_cell(0, line[0], img_bin_otsu.shape[1], line[1]-line[0]), config="--psm 12", lang='engorig'))
#     print("test")
#
#
#
#
# cv2.imshow("text lines", line_im)
# cv2.waitKey(0)



#download these pages as tiffs


print("test")