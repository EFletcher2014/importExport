import pytesseract
import cv2
import argparse
import numpy
import math
import pandas as pd
from scipy import stats

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image to be OCR'd")
args = vars(ap.parse_args())


# load the input image and convert it to grayscale
image = cv2.imread(args["image"])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_bin_otsu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
cv2.imshow('original', cv2.resize(image, None, fx=0.25, fy=0.25))
cv2.imshow('thresh', cv2.resize(img_bin_otsu, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

#extract tables
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, numpy.array(image).shape[1]//100))
eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (numpy.array(image).shape[1]//100, 1))
horizontal_lines = cv2.erode(img_bin_otsu, hor_kernel, iterations=5)
horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)

#search_line here if needed

#combine vertical and horizontal table lines into one image
vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
vertical_horizontal_lines = cv2.morphologyEx(vertical_horizontal_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#clean up horizontal lines in order to extend them as needed
horizontal_lines = cv2.erode(~horizontal_lines, kernel, iterations=3)
horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
horizontal_lines = ~horizontal_lines


cv2.imshow('hor', cv2.resize(horizontal_lines, None, fx=0.25, fy=0.25))
cv2.waitKey(0)


cv2.imshow('original lines', cv2.resize(vertical_horizontal_lines, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

#Identify where table is located, draw a bounding box on it since the report doesn't have one
line_indices = numpy.where(vertical_horizontal_lines == 0)
table_x = line_indices[1].min()
table_y = line_indices[0].min()
table_x1 = line_indices[1].max()
table_y1 = line_indices[0].max()
vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (table_x, table_y), (table_x1, table_y1), (0, 255, 0), 10)

#clean up vertical lines for extending as necessary
vertical_lines = cv2.erode(~vertical_lines, kernel, iterations=3)
vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
temp, vertical_lines = cv2.threshold(vertical_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


cv2.imshow('lines', cv2.resize(vertical_horizontal_lines, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

cv2.imshow('vert', cv2.resize(~vertical_lines, None, fx=0.25, fy=0.25))
cv2.waitKey(0)


#get individual vertical lines to extend as needed
vert_lines = cv2.findContours(~vertical_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
vert_lines = vert_lines[0] if len(vert_lines) == 2 else vert_lines[1]


#extend vertical lines
im = []
blank_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)

for line in vert_lines:
    line_end = line[:, 0, 1].max()
    x_start = line[:, 0, 0].min()
    x_end = line[:, 0, 0].max()

    #currently, only extends the lines closest to the bottom so that they're touching the bounding box
    if line_end > table_y1 - 25 and line_end < table_y1:
        im = cv2.rectangle(image, (x_start, line_end), (x_start+5, table_y1), (255, 255, 0), -1)
        vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (x_start, line_end), (x_start+5, table_y1), (0, 0, 0), -1)


#extend horizontal lines as needed
#add bounding box to vertical lines, since we will treat it as vertical lines
vertical_lines = cv2.rectangle(vertical_lines, (table_x, table_y), (table_x1, table_y1), (0, 0, 0), 10)

#separate individual horizontal lines for extension as needed
hor_lines = cv2.findContours(horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hor_lines = hor_lines[0] if len(hor_lines) == 2 else hor_lines[1]

for line in hor_lines:
    line_start = line[:, 0, 0].min()
    line_end = line[:, 0, 0].max()
    y = line[:, 0, 1].min()
    y1 = line[:, 0, 1].max()

    # find all vertical lines intersecting this row
    intersecting_lines = vertical_lines[y:y1, :]

    #extend start of lines if needed
    indices_start = numpy.where(intersecting_lines[:, 0:line_start] == 0)[1]
    if len(indices_start) > 0:
        line_start = indices_start[-1]
        im = cv2.rectangle(image, (line_start, y), (line[:, 0, 0].min(), y1), (0, 255, 0), -1)
        #cv2.imshow('added', cv2.resize(im, None, fx=0.25, fy=0.25))
        #cv2.waitKey(0)

    #extend end of lines if needed
    indices_end = numpy.where(intersecting_lines[:, line_end:] == 0)[1]
    if len(indices_end) > 0:
        line_end += indices_end[0]
        im = cv2.rectangle(image, (line_end - indices_end[0], y), (line_end, y1), (0, 255, 0), -1)
        #cv2.imshow('added', cv2.resize(im, None, fx=0.25, fy=0.25))
        #cv2.waitKey(0)

    #update main image of table boundaries to reflect extended lines
    vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (line_start, y), (line_end, y1), (0, 0, 0), -1)

#highlight areas where lines were extended, then show new lines
cv2.imshow('added', cv2.resize(im, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

cv2.imshow('extended lines', cv2.resize(vertical_horizontal_lines, None, fx=0.25, fy=0.25))
cv2.waitKey(0)


#isolate table's cells
cells = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cells = cells[0] if len(cells) == 2 else cells[1]


def parse_cells(x, y, edge, g_h, append_label, cols):

    if x >= edge:
        return cols
    else:
        grid = vertical_horizontal_lines[y:g_h, x:edge]
        cells = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cells = cells[0] if len(cells) == 2 else cells[1]

        if not cells:
            return cols

        # cv2.imshow('grid', image[y:g_h, x:edge])#cv2.resize(cls, None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)

        cell = cells[-1]
        cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cell)

        img = image[y+cell_y:y+cell_y+cell_h, x+cell_x:x+cell_x+cell_w]
        label = str(pytesseract.image_to_string(img, config="--psm 12", lang='engorig'))

        # cv2.imshow('cell', image[y+cell_y:y+cell_y+cell_h, x+cell_x:x+cell_x+cell_w])  # cv2.resize(cls, None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)

        #if cell isn't as tall as the entire section, know there are subheadings. Parse those instead
        if y + cell_y + cell_h < g_h - 10:
            append_label = label
            parse_cells(cell_x + x, cell_y + y + cell_h, cell_x + x + cell_w, g_h, append_label, cols)
            parse_cells(x + cell_x + cell_w, y + cell_y, edge, g_h, "", cols)
        else:
            cols.append(dict(start_x = cell_x + x, start_y = cell_y + y, height = cell_h,
                             width = cell_w, label = "\n".join([append_label, label]), data = []))
            parse_cells(x + cell_x + cell_w, y + cell_y, edge, g_h, append_label, cols)

        #move to next section
        #parse_cells(x + cell_x + cell_w, y + cell_y, edge, g_h, append_label, cols)
        return cols

cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cells[-1])
columns = parse_cells(table_x, table_y, table_x1, cell_y + cell_h, "", [])

#now that heading is extracted, need to extract data as well
no_lines = cv2.addWeighted(vertical_horizontal_lines, 0.5, img_bin_otsu, 0.5, 0.0)
#no_lines = cv2.erode(~no_lines, kernel, iterations=3)
#no_lines = cv2.morphologyEx(no_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
thresh, no_lines = cv2.threshold(no_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

first_col = no_lines[columns[0]["start_y"]+columns[0]["height"]:table_y1, columns[0]["start_x"]:columns[0]["start_x"]+columns[0]["width"]]

cv2.imshow("first_col", cv2.resize(first_col, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

# Dilate to combine adjacent text contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
dilate = cv2.dilate(first_col, kernel, iterations=2)

cv2.imshow("dilate", cv2.resize(dilate, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

# Find contours, highlight text areas, and extract ROIs
cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

parent_text_lines = []
child_text_lines = []
child = image.copy()
parent = image.copy()

for i in range(0, len(cnts)):
    c = cnts[i]
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    hier = hierarchy[0][i]

    if area > 500 and w > h and h > image.shape[0] / 200:
        # cv2.imshow("text line", cv2.resize(
        #     image[y + columns[0]["start_y"] + columns[0]["height"]:y + columns[0]["start_y"] + columns[0]["height"] + h,
        #     table_x:table_x1], None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)

        l_start = y + columns[0]["start_y"] + columns[0]["height"]
        l_end = y + columns[0]["start_y"] + columns[0]["height"] + h

        test = []

        # #if overlaps the previous row, ignore
        # if not text_lines or l_end <= text_lines[0][0]:
        #     test = cv2.rectangle(image, (table_x, l_start),
        #                          (table_x1, l_end),
        #                          color=(255, 0, 255), thickness=3)
        #     text_lines.insert(0, [l_start, l_end])
        #
        # test = cv2.rectangle(image, (table_x, l_start),
        #                      (table_x1, l_end),
        #                      color=(255, 0, 255), thickness=3)

        if hier[2] < 0:
            child_text_lines.insert(0, [l_start, l_end])
            child = cv2.rectangle(child, (table_x, l_start),
                             (table_x1, l_end),
                             color=(255, 0, 255), thickness=3)
            cv2.imshow("text line", cv2.resize(
                image[l_start:l_end,
                table_x:table_x1], None, fx=0.25, fy=0.25))
            cv2.waitKey(0)
        else:
            parent = cv2.rectangle(parent, (table_x, l_start),
                                 (table_x1, l_end),
                                 color=(255, 255, 0), thickness=3)
            parent_text_lines.insert(0, [l_start, l_end])

cv2.imshow("parent", cv2.resize(parent, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

cv2.imshow("child", cv2.resize(child, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

text_lines = []
temp = 0
for c in cnts:
    area = cv2.contourArea(c)
    x,y,w,h = cv2.boundingRect(c)
    hier = hierarchy[0][temp]

    if area > 500 and w > h and h > image.shape[0]/200:

        cv2.imshow("text line", cv2.resize(
            image[y + columns[0]["start_y"] + columns[0]["height"]:y + columns[0]["start_y"] + columns[0]["height"] + h,
            table_x:table_x1], None, fx=0.25, fy=0.25))
        cv2.waitKey(0)


        l_start = y + columns[0]["start_y"] + columns[0]["height"]
        l_end = y + columns[0]["start_y"] + columns[0]["height"] + h

        test = []

        # #if overlaps the previous row, ignore
        # if not text_lines or l_end <= text_lines[0][0]:
        #     test = cv2.rectangle(image, (table_x, l_start),
        #                          (table_x1, l_end),
        #                          color=(255, 0, 255), thickness=3)
        #     text_lines.insert(0, [l_start, l_end])
        #
        # test = cv2.rectangle(image, (table_x, l_start),
        #                      (table_x1, l_end),
        #                      color=(255, 0, 255), thickness=3)
        text_lines.insert(0, [l_start, l_end])
        temp += 1

line_height = stats.mode([line[1] - line[0] for line in text_lines])[0]

text_lines_height = [line for line in text_lines if line[1] - line[0] < line_height + 10 and line[1] - line[0] > line_height - 10]



for i in range(0, len(text_lines_height)):
    # #if overlaps the previous row, ignore
    if i == 0 or text_lines_height[i][0] >= text_lines_height[i-1][0]:
        cv2.imshow("text line", cv2.resize(
            image[line[0]:line[1],
            table_x:table_x1], None, fx=0.25, fy=0.25))
        cv2.waitKey(0)
        test = cv2.rectangle(image, (table_x, line[0]), (table_x1, line[1]), color = (255, 0, 255), thickness =  3)

cv2.imshow("text lines", cv2.resize(test, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

#loop through columns to collect data
for col in columns:
    col_im = no_lines[col["start_y"] + col["height"]:table_y1,
                col["start_x"]:col["start_x"] + col["width"]]
    # cv2.imshow("col", cv2.resize(col_im, None, fx=0.25, fy=0.25))
    # cv2.waitKey(0)

    for line in text_lines:
        cell_im = no_lines[line[0]:line[1], col["start_x"]:col["start_x"] + col["width"]]

        # cv2.imshow("row", cv2.resize(cell_im, None, fx=0.25, fy=0.25))
        # cv2.waitKey(0)
        text = str(pytesseract.image_to_string(cell_im, config="--psm 12", lang='engorig'))
        col["data"].append(text)

#write to CSV
labels = [x["label"] for x in columns]
data = ["" for i in range(0, len(columns))]
for x in range(0, len(columns)):
    data[x] = columns[x]["data"]
data = numpy.array(data).T.tolist()
#data.insert(0, labels)

df = pd.DataFrame(data = data)
df.to_csv(str(args["image"]).replace(".tif", ".csv"), index=False, header = labels)

#
# class column:
#     def __init__(self, start, end, label):
#         self.start = start
#         self.end = end
#         self.label = label
#         self.rows = []
#
# boxes = []
# cls = []
# x, y, w, h = cv2.boundingRect(cells[-1])
# header_h = y + h
# cols = []
# header_cells = 0
# header_rows = 0
# flag = False
# prev_x = 0
#
# for cell in reversed(cells):
#     x, y, w, h = cv2.boundingRect(cell)
#     img = image[y:y+h, x:x+h]
#     text = str(pytesseract.image_to_string(img, config="--psm 12", lang='engorig'))
#
#     if y < header_h-10 and not flag:
#         header_cells += 1
#         if x > prev_x:
#             if len(cols) > 0:
#                 cols[-1]["end"] = x
#             cols.append(dict(start = x, end = table_x1, label = text, data = ""))
#             prev_x = x
#
#             #if h < header_h - 10:
#
#
#             #parse_cells(x, y, w, h)
#
#         else:
#             prev_x = image.shape[1]
#     else:
#         flag = True
#
#
#
#     if True or (w<1000 and h<2900 and w>10 and h>10):
#         cls = cv2.rectangle(image,(x,y),(x+w,y+h),(255, 105, 180),2)
#         boxes.insert(0, [x,y,w,h])
#         cv2.imshow('cell', image[y:y+h, x:x+w])#cv2.resize(cls, None, fx=0.25, fy=0.25))
#         cv2.waitKey(0)
#
#
# cv2.imshow('box', cv2.resize(cls, None, fx=0.25, fy=0.25))
# cv2.waitKey(0)
#
# #use tesseract to read text from cells, then write to csv
# table = []
# cols = 0
# flag = False
# #loop through all boxes to run tesseract
# for box in boxes:
#     img = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
#     if not(flag):
#         if box[1] < boxes[0][1] + boxes[0][3]:
#             cols += 1
#         else:
#             flag = True
#
#     text = str(pytesseract.image_to_string(img, config="--psm 12", lang='engorig'))
#     print(text)
#     table.append(text)
#
# #write to CSV
# table = table + (cols - len(table) % cols) * [""]
# table = numpy.reshape(table, (-1, cols))
# df = pd.DataFrame(data=table)
# df.to_csv("1900pg188.csv")

