import pytesseract
import cv2
import argparse
import numpy
import math
import statistics
import pandas as pd
from scipy import stats
from textblob import TextBlob
import os
import string

SHOW_IMAGES = False

#cv2.findContours returns contours and hierarchy,
#   but we normally only want contours. Strip hierarchy
def handle_contours(cnts):
    return cnts[0] if len(cnts) == 2 else cnts[1]

#cv2.imshow requires multiple duplicate lines,
#   wrap it in this function to limit replication
def view(title, i, resize = False, pause = 100):
    if SHOW_IMAGES:
        if resize:
            i = cv2.resize(i, None, fx=0.25, fy=0.25)
        cv2.imshow(title, i)
        cv2.waitKey(pause)

#function to sort cells by order of top left corner
def sort_cells(cell):
    rect = cv2.boundingRect(cell)
    return rect[0] + rect[1]

#function to clean text pulled from inside cells
def clean_str(s, num_only = False):
    ret = str(s).replace("\n", "").replace("$", "").replace(",", "").replace(" ", "").replace("-", "")
    if not num_only:
        ret = ret.replace(".", "")
    return ret

#function to pull text from inside cells
def parse_text(cell_im, num_only = False, psm = 12):
    text = ""
    psm_str = "".join(["--psm ", str(psm)]) #will tell Tesseract what type of image to expect (text line, page, etc)
    if num_only:
        #if cell should contain data, expect it to only contain numbers
        result = pytesseract.image_to_data(cell_im, config=psm_str + " -c tessedit_char_whitelist=0123456789,", lang='engorig', output_type='data.frame')
        result = result[result.conf != -1]
        if len(result.conf) > 0:
            conf = min(result.conf)
        else:
            conf = -1
        #text currently split by word/line, so concatenate them. Then only keep digits
        text = (" ".join([str(int(r)) if isinstance(r, float) else str(r) for r in result.text.values])).replace(",", "")
        text = "".join([c for c in text if c.isdigit() or c == "."])
    else: #if not numbers (not expected to be data)
        result = pytesseract.image_to_data(cell_im, config=psm_str,
                                           lang='engorig', output_type='data.frame')
        result = result[result.conf != -1]
        if len(result.conf) > 0:
            conf = min(result.conf)
        else:
            conf = -1
        #text currently split by word/line, so concatenate them. Then only keep non-digits
        text = (" ".join([str(int(r)) if isinstance(r, float) else str(r) for r in result.text.values])).replace(",", "").replace(";", "")
        text = "".join([c for c in text if not c.isdigit()])

        if col == columns[0]: #if first column (countries), run through spell check
            tb = TextBlob(text)
            text = str(tb.correct())

    if text == "" and psm != 7: #if psm 12 retrieved no data, try again with psm of 7
        return parse_text(cell_im, num_only, 7)
    else:
        return text, conf


#function to remove noise from cell image
def clean_cell(x, y, w, h):
    #using dimensions, get image of just this cell
    im_to_clean = no_lines[y:y+h, x:x+w]

    #view("for cleaning", im_to_clean, 0)

    #dilate to isolate text contours from noise
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mega_dilation = cv2.dilate(im_to_clean, rect_kernel, iterations=1)

    #view("dilated", mega_dilation, 0)

    cell_im = im_to_clean.copy()

    #find contours in dilated image
    cnts = handle_contours(cv2.findContours(mega_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

    c_dims = [cv2.boundingRect(cnt) for cnt in cnts] #dimensions of each contour

    if len(c_dims) > 0:
        avg_w = statistics.mean([dim[2] for dim in c_dims])

    #loop through contours and draw over (erase) those that are too small to be characters
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # check if contour is at least 35px wide and 100px tall, and if
        # so, consider the contour a digit
        if w < avg_w/2 or w < 10 or h < 5:
            cell_im = cv2.drawContours(cell_im, [c], 0, (0, 0, 0), -1)

    #view("clean", cell_im, 0)

    return cell_im


#function to recursively parse columns from a heading
def parse_cells(x, y, edge, g_h, append_label, cols):

    if x >= edge: #if we've overstepped bounds of heading section, stop
        return cols
    else:

        #isolate all cells in this section
        grid = vertical_horizontal_lines[y:g_h, x:edge]
        cells = handle_contours(cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
        cells = [cell for cell in cells if cv2.contourArea(cell) > 2500]
        cells.sort(reverse=True, key=sort_cells)

        if not cells: #if no cells, stop
            return cols

        #view("grid", image[y:g_h, x:edge], 0)


        #Handle the top left most cell in this section
        cell = cells[-1]

        cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cell)
        img = clean_cell(x+cell_x, y+cell_y, cell_w, cell_h)
        label = str(pytesseract.image_to_string(img, config="--psm 12", lang='engorig')).replace(",", "").replace(";", "")

        if label == "": #if psm of 12 didn't retrieve any text, try again with psm 7
            label = str(pytesseract.image_to_string(img, config="--psm 7", lang='engorig')).replace(",", "").replace(";", "")

        #run text through spell checker
        tb = TextBlob(label)
        label = str(tb.correct())

        # if SHOW_IMAGES:
        #     view("cell", image[y+cell_y:y+cell_y+cell_h, x+cell_x:x+cell_x+cell_w], 0)
        #     cv2.rectangle(image, (x+cell_x, y+cell_y), (x+cell_x+cell_w, y+cell_y+cell_h), color = (255, 0, 255), thickness = 5)
        #     view("cell", image, True, 0)

        #if cell isn't as tall as the entire section, know there are subheadings. Parse those now
        if y + cell_y + cell_h < g_h - 10:
            #columns will only reflect the lowest row of heading, so append labels from columns in previous rows
            if append_label != "":
                label = "\n".join([append_label, label])
            append_label = label

            #parse cell below this, then move on to the rest of this section
            parse_cells(cell_x + x, cell_y + y + cell_h, cell_x + x + cell_w, g_h, append_label, cols)
            parse_cells(x + cell_x + cell_w, y + cell_y, edge, g_h, "", cols)
        else: #if no subheadings for this cell, and it's large enough to be a valid cell, add it to the list of columns
            if cell_w >= 80:
                cols.append(dict(start_x = cell_x + x, start_y = cell_y + y, height = cell_h,
                                width = cell_w, label = "\n".join([append_label, label]), data = [], confs = []))

            #then move on to the rest of the section
            parse_cells(x + cell_x + cell_w, y + cell_y, edge, g_h, append_label, cols)

        return cols

#function to delete lines that overlap
def handle_overlaps(line_in, comp_lines):
    #if no more lines, stop
    if line_in >= len(comp_lines):
        return comp_lines
    else:
        #retrieve lines that overlap with this line
        line = comp_lines[line_in]
        overlaps = [comp_line for comp_line in comp_lines if (line[0] > comp_line[0] and line[0] < comp_line[1]) or
                    (line[1] < comp_line[1] and line[1] > comp_line[0])]

        #if SHOW_IMAGES:
            # overlap = image[line[0]:line[1],
            #           table_x:table_x1].copy()
            # view(" ".join(["overlap: ", str(line[0]), str(line[1])]), overlap)

        if not overlaps: #if there are no overlapping lines
            #if line is tall enough to be split, split it
            if line[1] - line[0] >= (line_height * 2) - 10:
                end = line[1]
                line[1] = math.floor((end - line[0])/2) + line[0]
                comp_lines.insert(line_in + 1, [line[1], end])
            return handle_overlaps(line_in + 1, comp_lines)
        else: #if there are overlapping lines, delete or merge lines as needed
            return handle_overlaps(line_in, split(line_in, overlaps[0], comp_lines))

#function to delete or merge lines that overlap
def split(line_in, comp_line, comp_lines):
    new_lines = []
    line = comp_lines[line_in]

    # overlap creates three distinct lines
    new_lines.append([min(line[0], comp_line[0]), max(line[0], comp_line[0])])
    new_lines.append([max(line[0], comp_line[0]), min(line[1], comp_line[1])])
    new_lines.append([min(line[1], comp_line[1]), max(line[1], comp_line[1])])


    #if SHOW_IMAGES:
        # overlap = image[min(line[0], comp_line[0]): max(line[1], comp_line[1]),
        #     table_x:table_x1].copy()
        # overlap = cv2.rectangle(overlap, (0, line[0]-min(line[0], comp_line[0])), (table_x1, line[1] - min(line[0], comp_line[0])), color = (255, 0, 255), thickness = 5)
        # overlap = cv2.rectangle(overlap, (0, comp_line[0] - min(line[0], comp_line[0])), (table_x1, comp_line[1] - min(line[0], comp_line[0])), color = (255, 255, 0), thickness = 3)
        # view(" ".join(["overlap: ", str(line[0]), str(line[1]), str(comp_line[0]), str(comp_line[1])]), overlap)

    #if new lines are big enough to stand alone, let them. Otherwise merge
    l = 0
    while l < len(new_lines):
        li = new_lines[l]
        if li[1] - li[0] <= line_height - 6:
            if l < len(new_lines)-1:
                new_lines[l+1][0] = li[0]
            else:
                new_lines[l - 1][1] = li[1]

            new_lines.remove(li)
        else:
            l += 1
    comp_lines.remove(line)
    comp_lines.remove(comp_line)

    for x in range(0, len(new_lines)):
        comp_lines.insert(line_in + x, new_lines[x])
    comp_lines.sort()
    return comp_lines

###Code starts here###
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
    help="path to input directory to be OCR'd")
args = vars(ap.parse_args())

#load files
files = os.listdir(args["directory"])

#prepare output directory for end of program
output_directory = args["directory"].replace(args["directory"].split("/")[-1], "images_as_csv")


#loop through files in input directory to run program
for file in files:
    print(" ".join(["attempting", file]))
    file_path = "/".join([args["directory"], file])
    image = cv2.imread(file_path)

    #image preprocessing
    # color over any phantom lines that may have appeared at bottom and top of page from scanning
    image = cv2.rectangle(image, (0, 0), (numpy.array(image).shape[1], 30), (255, 255, 255), -1)
    image = cv2.rectangle(image, (0, numpy.array(image).shape[0] - 50),
                          (numpy.array(image).shape[1], numpy.array(image).shape[0]), (255, 255, 255), -1)

    #convert image to grayscale and then threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_bin_otsu = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)
    view("original", image, True)

    #extract table
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, numpy.array(image).shape[1]//100))
    eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (numpy.array(image).shape[1]//100, 1))
    horizontal_lines = cv2.erode(img_bin_otsu, hor_kernel, iterations=5)
    horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)

    #combine vertical and horizontal table lines into one image representing table grid
    vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
    vertical_horizontal_lines = cv2.morphologyEx(vertical_horizontal_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #create copy of image with no table lines
    no_lines = cv2.addWeighted(vertical_horizontal_lines, 0.5, img_bin_otsu, 0.5, 0.0)
    thresh, no_lines = cv2.threshold(no_lines, 128, 255,
                                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    view("no lines", no_lines, True)

    #clean up horizontal lines in order to extend them as needed
    horizontal_lines = cv2.erode(~horizontal_lines, kernel, iterations=3)
    horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    horizontal_lines = ~horizontal_lines


    view("hor", horizontal_lines, True)

    #view("original lines", vertical_horizontal_lines, True)

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

    view("lines", vertical_horizontal_lines, True)
    view("vert", ~vertical_lines, True)

    #get individual vertical lines to extend as needed
    vert_lines = handle_contours(cv2.findContours(~vertical_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

    #extend vertical lines
    blank_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)
    extended_lines = image.copy()

    #view("rotated", rotated, True)

    #add bounding box to horizontal lines, since we will treat it as horizontal lines
    horizontal_lines = ~horizontal_lines
    horizontal_lines = cv2.rectangle(horizontal_lines, (table_x, table_y), (table_x1, table_y1), (0, 0, 0), 10)

    #view("hor", horizontal_lines, True)

    #loop through vertical lines to extend as needed
    for line in vert_lines:
        line_start = line[:, 0, 1].min() #vertical points where line starts and ends
        line_end = line[:, 0, 1].max()
        x_start = line[numpy.where(line[:, 0, 1] == line_start), 0, 0].min() #horizontal points where line starts and ends
        x_end = line[numpy.where(line[:, 0, 1] == line_end), 0, 0].max()    #   (line is a polygon, not a true line)

        #get slope of line
        M = cv2.moments(line)
        theta = 0.5 * numpy.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"])
        angle = 180 / numpy.pi * theta
        slope = numpy.tan(theta)

        #extend the lines closest to the bottom so that they're touching the bounding box of the table
        if line_end > table_y1 - 25 and line_end < table_y1:
            im = cv2.rectangle(image, (x_start, line_end), (x_start+5, table_y1), (255, 255, 0), -1)
            vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (x_start, line_end), (x_start+5, table_y1), (0, 0, 0), -1)

        # find all horizontal lines intersecting this column
        intersecting_lines = vertical_horizontal_lines[:, x_start-7:x_end+7]

        # view("line", vertical_lines[:, x_start-5:x_end+5], True)
        # view("intersect", vertical_horizontal_lines[:, x_start-5:x_end+5], True, 0)

        # extend start of lines if needed
        indices_start = numpy.where(intersecting_lines[0:line_start, :] == 0)[0] #indices where intersecting horizontal lines start
        if len(indices_start) > 0: #if there are intersecting lines above where this line starts, extend the line
            extended_lines = cv2.line(extended_lines, (x_start - math.floor((line_start - indices_start[-1]) / slope), indices_start[-1]),
                                      (x_end, line_end), (0, 255, 0), 3)
            line_start = indices_start[-1]
            # view("added", extended_lines, True)

        # extend end of lines if needed
        indices_end = numpy.where(intersecting_lines[line_end:, :] == 0)[0] #indices where intersecting horizontal lines end
        if len(indices_end) > 0: #if there are intersecting lines below where this line ends, extend the line
            line_end += indices_end[0]
            extended_lines = cv2.line(extended_lines, (x_start, line_start - indices_end[0]), (x_end, line_end), (0, 255, 0), 3)
            #view("added", extended_lines, True)

        # update main image of table to reflect extended lines
        vertical_horizontal_lines = cv2.line(vertical_horizontal_lines, (x_start, line_start),
                                             (x_end, line_end), (0, 0, 0), 3)
        vertical_lines = cv2.line(vertical_lines, (x_start, line_start),
                                             (x_end, line_end), (0, 0, 0), 3)

    #view("added", extended_lines, True)
    #view("extended lines vert", vertical_horizontal_lines, True)


    #extend horizontal lines as needed
    #add bounding box to vertical lines, since we will treat it as vertical lines
    vertical_lines = cv2.rectangle(vertical_lines, (table_x, table_y), (table_x1, table_y1), (0, 0, 0), 10)
    #view("vert_lines", vertical_lines, True)

    #add bounding box to horizontal lines since we will treat those as horizontal lines
    horizontal_lines = cv2.rectangle(horizontal_lines, (table_x, table_y), (table_x1, table_y1), (255, 255, 255), 10)
    #view("hor_lines1", horizontal_lines, True)

    #isolate individual horizontal lines, loop through to extend
    hor_lines = handle_contours(cv2.findContours(~horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
    for line in hor_lines:
        line_start = line[:, 0, 0].min() #horizontal point where line starts and ends
        line_end = line[:, 0, 0].max()
        y = line[:, 0, 1].min() #vertical point where line starts and ends
        y1 = line[:, 0, 1].max()    #   (line is a polygon, not a true line)

        # find all vertical lines intersecting this row
        intersecting_lines = vertical_lines[y:y1, :]

        #extend start of lines if needed
        indices_start = numpy.where(intersecting_lines[:, 0:line_start] == 0)[1] #indices where intersecting lines start
        if len(indices_start) > 0: #if there are intersecting vertical lines to the left of this line, extend
            line_start = indices_start[-1]
            extended_lines = cv2.rectangle(extended_lines, (line_start, y), (line[:, 0, 0].min(), y1), (0, 255, 0), -1)
            #view("added", im, True)

        #extend end of lines if needed
        indices_end = numpy.where(intersecting_lines[:, line_end:] == 0)[1] #indices where intersecting lines end
        if len(indices_end) > 0: #if there are intersecting vertical lines to the right of this line, extend
            line_end += indices_end[0]
            extended_lines = cv2.rectangle(extended_lines, (line_end - indices_end[0], y), (line_end, y1), (0, 255, 0), -1)
            #view("added", im, True)

        #update main image of table boundaries to reflect extended lines
        vertical_horizontal_lines = cv2.rectangle(vertical_horizontal_lines, (line_start, y), (line_end, y1), (0, 0, 0), -1)

    #view("added", extended_lines, True)
    view("extended lines", vertical_horizontal_lines, True)


    #parse columns of table
    #using table grid, isolate cells
    cells = handle_contours(cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
    cells = [cell for cell in cells if cv2.contourArea(cell) > 2500]
    cells.sort(reverse=True, key=sort_cells)

    #loop through cells, starting in top left corner, until we find one that might be a column label
    cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cells[-2]) #findContours also includes large boundary contours,
                                                                    # so ignore those
    cell_in = len(cells) - 2
    cell_h = 501

    #loop through cells until we find one that's the appropriate height for a heading
    while cell_in in range(0, len(cells)) and cell_h > 500:
        cell_x, cell_y, cell_w, cell_h = cv2.boundingRect(cells[cell_in])
        cell_in -= 1
    #view("cell", image[cell_y:cell_y + cell_h + 66, cell_x:cell_x + cell_w], True)

    #1900 report has a small column on the left with indices
    col1_width = cv2.boundingRect(cells[-1])[2]

    #In 1900 report, expect one more heading that doesn't have a line below it--draw this line
    #view("final heading", image[table_y:cell_y+cell_h+66, table_x:table_x1], True)
    vertical_horizontal_lines = cv2.line(vertical_horizontal_lines, (table_x + col1_width, cell_y + cell_h + 66), (table_x1, cell_y + cell_h + 66), (0, 0, 0), 3)
    #view("heading", vertical_horizontal_lines, True)


    #Now that we know the boundaries of the heading, extract columns from this section of image
    columns = parse_cells(table_x, table_y, table_x1, cell_y + cell_h + 66, "", [])

    #remove columns parsed incorrectly (not wide enough to contain text or contain no text)
    columns = [col for col in columns if col["width"] > 100 or clean_str(col["label"]) != ""]



    #now that columns are extracted, extract the data associated with them
    first_col = no_lines[columns[0]["start_y"]+columns[0]["height"]:table_y1, columns[0]["start_x"]:columns[0]["start_x"]+columns[0]["width"]]
    view("first_col", first_col, True)

    #using first column, dilate to identify text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(first_col, kernel, iterations=2)
    view("dilate", dilate, True)

    #contours should represent text lines
    cnts = handle_contours(cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))

    #get all possible lines
    text_lines = []
    for i in range(0, len(cnts)): #loop through contours to see if they might represent a line
        c = cnts[i]
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if area > 200 and w > h and h > image.shape[0] / 300: #if contour is big enough to be a line
            #view("text line", image[y + columns[0]["start_y"] + columns[0]["height"]:y + columns[0]["star
            #     t_y"] + columns[0]["height"] + h,
            #     table_x:table_x1], True)

            l_start = y + columns[0]["start_y"] + columns[0]["height"]
            l_end = y + columns[0]["start_y"] + columns[0]["height"] + h
            text_lines.insert(0, [l_start, l_end]) #add line to list of lines

    #create image of text lines for user validation
    # if SHOW_IMAGES:
        # line_im = image.copy()
        # for line in text_lines:
        #     line_im = cv2.rectangle(line_im, (table_x, line[0]), (table_x1, line[1]), color = (255, 255, 0), thickness = 3)
        #view("text lines", line_im, True)

    #calculate suspected line height using mode, remove lines that are too short
    line_height = stats.mode([line[1] - line[0] for line in text_lines])[0]
    text_lines_no_short = [line for line in text_lines if line[1] - line[0] >= line_height - 2]

    #remove duplicate lines
    text_lines_no_dup = []
    for line in text_lines_no_short:
        if line not in text_lines_no_dup:
            text_lines_no_dup.append(line)

    text_lines_no_overlap = handle_overlaps(0, text_lines_no_dup)
    text_lines_no_overlap.sort()

    if SHOW_IMAGES:
        line_im = image.copy()
        for line in text_lines_no_overlap:
            line_im = cv2.rectangle(line_im, (table_x, line[0]), (table_x1, line[1]), color = (255, 255, 0), thickness = 3)
        view("text lines", line_im, True)



    #now know where lines and columns should be, so loop through to extract data
    columns[0]["start_x"] -= 100 #add a buffer to first column
    columns[0]["width"] += 100
    for col in columns:
        col_im = no_lines[col["start_y"] + col["height"]:table_y1,
                    col["start_x"]:col["start_x"] + col["width"]] #get image of a given column

        for line in text_lines_no_overlap: #loop through text lines in the column
            cell_im = clean_cell(col["start_x"], line[0], col["width"], line[1]-line[0]) #cell represents one text line
                                                                                        #  in a given column
            text, conf = parse_text(cell_im, num_only=(col != columns[0])) #extract text from cell
            col["data"].append(text) #save text and confidence to the column
            col["confs"].append(conf)

    #prepare data to write to CSV
    labels = [x["label"] for x in columns] #labels will be used to make header in csv
    for l in range(0, len(labels)): #confidence values will be added as a column after the column they reference
        labels.insert(2*l+1, "".join(["col", str(l+1), "_CONF"]))

    data = ["" for i in range(0, 2*len(columns))]
    for x in range(0, len(columns)): #for each column, extract data and add to list, add confidence in adjacent column
        data[x*2] = columns[x]["data"]
        data[2*x+1] = columns[x]["confs"]
    table = numpy.array(data).T.tolist()

    #Clean data
    # first: find cells that are likely to contain relevant data, ignoring confidence values
    contain_data = [[clean_str(row[c]).isnumeric() and c%2 == 0 for c in range(0, len(row))] for row in table]
    contain_data_row = [any(row) for row in contain_data]
    contain_data_col = [any([row[c] for row in contain_data]) for c in range(0, len(contain_data[0]))]
    data = [table[r] for r in range(0, len(table)) if contain_data_row[r] == True]

    clean_data = [["" for c in row] for row in table]
    for r in range(0, len(table)):
        for c in range(0, len(table[r])):
            if contain_data_col[c]:
                clean_data[r][c] = clean_str(table[r][c], c!=0)
            else:
                if not any(char.isalpha() for char in table[r][c]):
                    clean_data[r][c] = str(table[r][c])
                else:
                    clean_data[r][c] = str(table[r][c]).replace(".", "")
    df = pd.DataFrame(data = clean_data)

    #write to CSV
    isExist = os.path.exists(output_directory)
    if not isExist:
        os.makedirs(output_directory)
    df.to_csv("/".join([output_directory, file.replace(".tiff", ".csv")]), index=False, header = [str(row).translate(str.maketrans('', '', string.punctuation)) for row in labels])

    #if user wants to see images, wait for input to close them
    if SHOW_IMAGES:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

