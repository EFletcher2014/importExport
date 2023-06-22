#### Save this as dual_input.py ####
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import Tk, END, Text, TOP, BOTH, X, Y, N, LEFT, RIGHT
from tkinter.ttk import Frame, Label, Entry, Button
from PIL import ImageTk, Image
import math

# Good habit to put your GUI in a class to make it self-contained
class SimpleDialog(Frame):

    def __init__(self, im):
        super().__init__()
        # self allow the variable to be used anywhere in the class
        self.output1 = ""
        self.output2 = ""
        self.image = im
        self.initUI()

    def initUI(self):

        self.master.title("Which pages?")
        self.pack(fill=BOTH, expand=True)

        # Create an object of tkinter ImageTk
        img = Image.open(self.image)
        img = img.resize([math.floor(img.width/8), math.floor(img.height/8)])
        img = ImageTk.PhotoImage(img)

        # Create a Label Widget to display the text or Image
        frame4 = Frame(self)
        frame4.pack(fill = Y)
        label = Label(frame4, image=img, width=3000)
        label.image = img
        label.pack(side = "bottom", fill = BOTH, expand=True)

        frame1 = Frame(self)
        frame1.pack(fill=X)

        lbl1 = Label(frame1, text="Start:", width=6)
        lbl1.pack(side=LEFT, padx=5, pady=10)

        self.entry1 = Entry(frame1, textvariable=self.output1)
        self.entry1.pack(fill=X, padx=5, expand=True)

        frame2 = Frame(self)
        frame2.pack(fill=X)

        lbl2 = Label(frame2, text="End:", width=6)
        lbl2.pack(side=LEFT, padx=5, pady=10)

        self.entry2 = Entry(frame2)
        self.entry2.pack(fill=X, padx=5, expand=True)

        frame3 = Frame(self)
        frame3.pack(fill=X)

        # Command tells the form what to do when the button is clicked
        btn = Button(frame3, text="Submit", command=self.onSubmit)
        btn.pack(padx=5, pady=10)

    def onSubmit(self):

        self.output1 = self.entry1.get()
        self.output2 = self.entry2.get()
        self.quit()

def main(in1, in2, im):

    # This part triggers the dialog
    root = Tk()
    root.geometry("1000x1000+300+300")
    app = SimpleDialog(im)
    app.entry1.insert(END, str(in1))
    app.entry2.insert(END, str(in2))


    root.mainloop()

    # Here we can act on the form components or
    # better yet, copy the output to a new variable
    user_input = (app.output1, app.output2)
    # Get rid of the error message if the user clicks the
    # close icon instead of the submit button
    # Any component of the dialog will no longer be available
    # past this point
    try:
        root.destroy()
    except:
        pass
    # To use data outside of function
    # Can either be used in __main__
    # or by external script depending on
    # what calls main()
    return user_input

# Allow dialog to run either as a script or called from another program
if __name__ == '__main__':
    follow_on_variable = main("274", "250", "/home/emily/Downloads/1900/temp_images/mdp.39015006977725page270.tiff")
    # This shows the outputs captured when called directly as `python dual_input.py`
    print(follow_on_variable)
#### End of dual_input.py code dialog code file ####

### Example of using from code file as opposed to ###
### calling directly                              ###
### Save this as i_do_work_here.py                ###
#! /usr/bin/env python
# -*- coding: utf-8 -*-

