import ensemble
from tkinter import *
from PIL import ImageTk, Image
import os

from tkinter import *

accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP = ensemble.startup()

print(predictionsENS[0])
count = 0

def getStats(count,accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP):
    DNNprediction = "Prediction: "
    if predictionsDNN[count][0] >= predictionsDNN[count][1] and predictionsDNN[count][0]>=predictionsDNN[count][2]:
        DNNprediction+= "Healthy"
    elif predictionsDNN[count][1] >= predictionsDNN[count][0] and predictionsDNN[count][1]>=predictionsDNN[count][2]:
        DNNprediction+= "Medium"
    elif predictionsDNN[count][2] >= predictionsDNN[count][0] and predictionsDNN[count][2]>=predictionsDNN[count][1]:
        DNNprediction+= "High"
    CNNprediction = "Prediction: "
    if predictionsCNN[count][0] >= predictionsCNN[count][1] and predictionsCNN[count][0]>=predictionsCNN[count][2]:
        CNNprediction+= "Healthy"
    elif predictionsCNN[count][1] >= predictionsCNN[count][0] and predictionsCNN[count][1]>=predictionsCNN[count][2]:
        CNNprediction+= "Medium"
    elif predictionsCNN[count][2] >= predictionsCNN[count][0] and predictionsCNN[count][2]>=predictionsCNN[count][1]:
        CNNprediction+= "High"
    TREEprediction = "Prediction: " 
    if predictionsTREE[count][0] >= predictionsTREE[count][1] and predictionsTREE[count][0]>=predictionsTREE[count][2]:
        TREEprediction+= "Healthy"
    elif predictionsTREE[count][1] >= predictionsTREE[count][0] and predictionsTREE[count][1]>=predictionsTREE[count][2]:
        TREEprediction+= "Medium"
    elif predictionsTREE[count][2] >= predictionsTREE[count][0] and predictionsTREE[count][2]>=predictionsTREE[count][1]:
        TREEprediction+= "High"
    SVMprediction = "Prediction: "
    if predictionsSVM[count][0] >= predictionsSVM[count][1] and predictionsSVM[count][0]>=predictionsSVM[count][2]:
        SVMprediction+= "Healthy"
    elif predictionsSVM[count][1] >= predictionsSVM[count][0] and predictionsSVM[count][1]>=predictionsSVM[count][2]:
        SVMprediction+= "Medium"
    elif predictionsSVM[count][2] >= predictionsSVM[count][0] and predictionsSVM[count][2]>=predictionsSVM[count][1]:
        SVMprediction+= "High"
    ENSprediction = "Ensemble Prediction: "
    if predictionsENS[count] == 0:
        ENSprediction+= "Healthy"
    elif predictionsENS[count] == 1:
        ENSprediction+= "Medium"
    elif predictionsENS[count] == 2:
        ENSprediction+= "High"
    EXPprediction = "Actual: "
    if predictionsEXP[count] == 0:
        EXPprediction+= "Healthy"
    elif predictionsEXP[count] == 1:
        EXPprediction+= "Medium"
    elif predictionsEXP[count] == 2:
        EXPprediction+= "High"
    DNNacc = "Accuracy: " + ((str(accuracyDNN))[:5]) + "%"
    CNNacc = "Accuracy: " +((str(accuracyCNN))[:5]) + "%"
    TREEacc = "Accuracy: " +((str(accuracyTREE))[:5])+ "%"
    SVMacc = "Accuracy: " +((str(accuracySVM))[:5])+ "%"
    DNNconf = "Confidence: " + (str(max(predictionsDNN[count]))[:5]) + "%"
    CNNconf = "Confidence: " + (str(max(predictionsCNN[count]))[:5]) + "%"
    SVMconf = "Confidence: " + (str(max(predictionsSVM[count]))[:5]) + "%"
    TREEconf = "Confidence: " + (str(max(predictionsTREE[count]))[:5]) +"%"
    sampleName = "Sample #" + str(count+1)
    return ENSprediction, DNNprediction, CNNprediction, TREEprediction, SVMprediction, DNNacc, CNNacc, TREEacc, SVMacc, DNNconf, CNNconf,SVMconf, TREEconf, EXPprediction, sampleName

ENSprediction, DNNprediction, CNNprediction, TREEprediction, SVMprediction, DNNacc, CNNacc, TREEacc, SVMacc, DNNconf, CNNconf,SVMconf, TREEconf, EXPprediction, sampleName = getStats(count, accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP)


def callback(event):
    global sampleNum

    if (event.x >= 445 and event.x <= 500 and event.y >= 0 and event.y <= 50):
        sampleNum = sampleNum + 1
        w.itemconfig(titleText, text="Sample #" + str(sampleNum))
    if (event.x >= 0 and event.x <= 50 and event.y >= 0 and event.y <= 50):
        sampleNum = sampleNum - 1
        w.itemconfig(titleText, text="Sample #" + str(sampleNum))

master = Tk()

sampleNum = 1

w = Canvas(master, width=500, height=500)
w.bind("<Button-1>", callback)
w.pack()
w.create_rectangle(55,2, 445, 50, fill = "grey") 
w.create_line(0,50,500,50,width = 3) 
titleText = w.create_text(250,25, text="Sample #" + str(sampleNum))
w.create_line(0,275,500,275,width = 3)
w.create_line(250,275,250,500, width = 3)
w.create_text(125,290, text="Spectra:")
w.create_text(375,290, text="Key Features:")
if predictionsENS[count] == predictionsEXP[count]:
    w.create_rectangle(0,225, 500, 275, fill = "#98FB98")  
else:
    w.create_rectangle(0,225, 500, 275, fill = "pink")  
w.create_text(125,250, text = ENSprediction)
w.create_text(375,250, text = EXPprediction)
w.create_line(125,50,125,225, width = 3)
w.create_line(250,50,250,225, width = 3)
w.create_line(375,50,375,225, width = 3)
w.create_line(0,225,500,225, width = 3)

#arrow left
w.create_line(15, 25, 40, 10, width = 3)
w.create_line(15, 25, 40, 40, width = 3)

#arrow right
w.create_line(460, 10, 485, 25, width = 3)
w.create_line(460, 40, 485, 25, width = 3)

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


# img1 = ImageTk.PhotoImage(Image.open("doggie.png").resize((244, 195)))
# img2 = ImageTk.PhotoImage(Image.open("cathat.png").resize((247, 195)))

# w.create_image(0,305,anchor = NW, image = img1)
# w.create_image(256,305, anchor = NW, image = img2)

img1 = ImageTk.PhotoImage(Image.open("doggie.png").resize((244, 195)))
img2 = ImageTk.PhotoImage(Image.open("cathat.png").resize((247, 195)))

w.create_image(0,305,anchor = NW, image = img1)
w.create_image(256,305, anchor = NW, image = img2)

mainloop()