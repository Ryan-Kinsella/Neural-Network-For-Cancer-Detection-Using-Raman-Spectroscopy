from tkinter import *

master = Tk()

value = 2

w = Canvas(master, width=500, height=500)
w.pack()
w.create_rectangle(0,2, 500, 50, fill = "grey")  
w.create_text(250,25, text="Sample #1")
w.create_line(0,275,500,275,width = 3)
w.create_line(250,275,250,500, width = 3)
w.create_text(125,290, text="Spectra:")
w.create_text(375,290, text="Key Features:")
w.create_text(250,250, text ="Ensemble Prediction: Non-cancerous",)
w.create_line(125,50,125,225, width = 3)
w.create_line(250,50,250,225, width = 3)
w.create_line(375,50,375,225, width = 3)
w.create_line(0,225,500,225, width = 3)
w.create_text(62,65, text = "DNN:")



w.create_text(187,65, text = "CNN:")

w.create_text(312,65, text = "SVM:")

w.create_text(437,65, text = "TREE:")

mainloop()
