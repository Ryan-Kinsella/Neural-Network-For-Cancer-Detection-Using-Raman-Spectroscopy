import ensemble
from tkinter import *
from PIL import ImageTk, Image
import os

accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE = ensemble.startup()
print(accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE)


DNNprediction = "Prediction: "
CNNprediction = "Prediction: "
TREEprediction = "Prediction: " 
SVMprediction = "Prediction: "
DNNacc = "Accuracy: " + ((str(accuracyDNN))[:5]) + "%"
CNNacc = "Accuracy: " +((str(accuracyCNN))[:5]) + "%"
TREEacc = "Accuracy: " +((str(accuracyTREE))[:5])+ "%"
SVMacc = "Accuracy: " +((str(accuracySVM))[:5])+ "%"
DNNconf = "Confidence: " + "%"
CNNconf = "Confidence: " + "%"
SVMconf = "Confidence: " + "%"
TREEconf = "Confidence: " + "%"


master = Tk()


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
w.create_text(62,100, text = DNNprediction)
w.create_text(62,135, text = DNNacc)
w.create_text(62,170, text = DNNconf)


w.create_text(187,65, text = "CNN:")
w.create_text(187,100, text = CNNprediction)
w.create_text(187,135, text = CNNacc)
w.create_text(187,170, text = CNNconf)

w.create_text(312,65, text = "SVM:")
w.create_text(312,100, text = SVMprediction)
w.create_text(312,135, text = SVMacc)
w.create_text(312,170, text = SVMconf)


w.create_text(437,65, text = "TREE:")
w.create_text(437,100, text = TREEprediction)
w.create_text(437,135, text = TREEacc)
w.create_text(437,170, text = TREEconf)

img1 = ImageTk.PhotoImage(Image.open("doggie.png").resize((244, 195)))
img2 = ImageTk.PhotoImage(Image.open("cathat.png").resize((247, 195)))

w.create_image(0,305,anchor = NW, image = img1)
w.create_image(256,305, anchor = NW, image = img2)

mainloop()
