import pytesseract
import cv2
import argparse
import numpy
import math
import pandas as pd

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
x = line_indices[1].min()
y = line_indices[0].min()
x1 = line_indices[1].max()
y1 = line_indices[0].max()
vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (x, y), (x1, y1), (0, 255, 0), 10)

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
    if line_end > y1 - 25 and line_end < y1:
        im = cv2.rectangle(image, (x_start, line_end), (x_start+5, y1), (255, 255, 0), -1)
        vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (x_start, line_end), (x_start+5, y1), (0, 0, 0), -1)


#extend horizontal lines as needed
#add bounding box to vertical lines, since we will treat it as vertical lines
vertical_lines = cv2.rectangle(vertical_lines, (x, y), (x1, y1), (0, 0, 0), 10)

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
contours = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

boxes = []
cnts = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (w<1000 and h<2900 and w>10 and h>10):
        cnts = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.insert(0, [x,y,w,h])


cv2.imshow('box', cv2.resize(cnts, None, fx=0.25, fy=0.25))
cv2.waitKey(0)

#use tesseract to read text from cells, then write to csv
table = []
cols = 0
flag = False
#loop through all boxes to run tesseract
for box in boxes:
    img = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    if not(flag):
        if box[1] < boxes[0][1] + boxes[0][3]:
            cols += 1
        else:
            flag = True

    text = str(pytesseract.image_to_string(img, config="--psm 12", lang='engorig'))
    print(text)
    table.append(text)

#write to CSV
table = table + (cols - len(table) % cols) * [""]
table = numpy.reshape(table, (-1, cols))
df = pd.DataFrame(data=table)
df.to_csv("1900pg188.csv")

