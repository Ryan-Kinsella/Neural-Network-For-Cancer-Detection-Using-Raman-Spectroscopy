import ensemble
from tkinter import *   
import os

accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP = ensemble.startup()
sampleNum = 0

def getStats(count,accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP):
    DNNprediction = "Prediction: "
    if predictionsDNN[count][0] >= predictionsDNN[count][1] and predictionsDNN[count][0]>=predictionsDNN[count][2]:
        DNNprediction+= "Healthy"
    elif predictionsDNN[count][1] >= predictionsDNN[count][0] and predictionsDNN[count][1]>=predictionsDNN[count][2]:
        DNNprediction+= "Low"
    elif predictionsDNN[count][2] >= predictionsDNN[count][0] and predictionsDNN[count][2]>=predictionsDNN[count][1]:
        DNNprediction+= "High"
    CNNprediction = "Prediction: "
    if predictionsCNN[count][0] >= predictionsCNN[count][1] and predictionsCNN[count][0]>=predictionsCNN[count][2]:
        CNNprediction+= "Healthy"
    elif predictionsCNN[count][1] >= predictionsCNN[count][0] and predictionsCNN[count][1]>=predictionsCNN[count][2]:
        CNNprediction+= "Low"
    elif predictionsCNN[count][2] >= predictionsCNN[count][0] and predictionsCNN[count][2]>=predictionsCNN[count][1]:
        CNNprediction+= "High"
    TREEprediction = "Prediction: " 
    if predictionsTREE[count][0] >= predictionsTREE[count][1] and predictionsTREE[count][0]>=predictionsTREE[count][2]:
        TREEprediction+= "Healthy"
    elif predictionsTREE[count][1] >= predictionsTREE[count][0] and predictionsTREE[count][1]>=predictionsTREE[count][2]:
        TREEprediction+= "Low"
    elif predictionsTREE[count][2] >= predictionsTREE[count][0] and predictionsTREE[count][2]>=predictionsTREE[count][1]:
        TREEprediction+= "High"
    SVMprediction = "Prediction: "
    if predictionsSVM[count][0] >= predictionsSVM[count][1] and predictionsSVM[count][0]>=predictionsSVM[count][2]:
        SVMprediction+= "Healthy"
    elif predictionsSVM[count][1] >= predictionsSVM[count][0] and predictionsSVM[count][1]>=predictionsSVM[count][2]:
        SVMprediction+= "Low"
    elif predictionsSVM[count][2] >= predictionsSVM[count][0] and predictionsSVM[count][2]>=predictionsSVM[count][1]:
        SVMprediction+= "High"
    ENSprediction = "Ensemble Prediction: "
    if predictionsENS[count] == 0:
        ENSprediction+= "Healthy"
    elif predictionsENS[count] == 1:
        ENSprediction+= "Low"
    elif predictionsENS[count] == 2:
        ENSprediction+= "High"
    EXPprediction = "Actual: "
    if predictionsEXP[count] == 0:
        EXPprediction+= "Healthy"
    elif predictionsEXP[count] == 1:
        EXPprediction+= "Low"
    elif predictionsEXP[count] == 2:
        EXPprediction+= "High"
    DNNacc = "Accuracy: " + ((str(accuracyDNN*100))[:5]) + "%"
    CNNacc = "Accuracy: " +((str(accuracyCNN*100))[:5]) + "%"
    TREEacc = "Accuracy: " +((str(accuracyTREE*100))[:5])+ "%"
    SVMacc = "Accuracy: " +((str(accuracySVM*100))[:5])+ "%"
    DNNconf = "Confidence: " + (str(max(predictionsDNN[count])*100)[:5]) + "%"
    CNNconf = "Confidence: " + (str(max(predictionsCNN[count])*100)[:5]) + "%"
    SVMconf = "Confidence: " + (str(max(predictionsSVM[count])*100)[:5]) + "%"
    TREEconf = "Confidence: " + (str(max(predictionsTREE[count])*100)[:5]) +"%"
    sampleName = "Sample #" + str(sampleNum+1)
    return ENSprediction, DNNprediction, CNNprediction, TREEprediction, SVMprediction, DNNacc, CNNacc, TREEacc, SVMacc, DNNconf, CNNconf,SVMconf, TREEconf, EXPprediction, sampleName

ENSprediction, DNNprediction, CNNprediction, TREEprediction, SVMprediction, DNNacc, CNNacc, TREEacc, SVMacc, DNNconf, CNNconf,SVMconf, TREEconf, EXPprediction, sampleName = getStats(sampleNum, accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP)


def callback(event):
    global sampleNum

    if (event.x >= 445 and event.x <= 500 and event.y >= 0 and event.y <= 50):
        sampleNum = sampleNum + 1
    if (event.x >= 0 and event.x <= 50 and event.y >= 0 and event.y <= 50):
        sampleNum = sampleNum - 1
    if (sampleNum < 0):
        sampleNum = len(predictionsENS)-1  
    if (sampleNum > len(predictionsENS)-1):
        sampleNum = 0  
    
    
    ENSprediction, DNNprediction, CNNprediction, TREEprediction, SVMprediction, DNNacc, CNNacc, TREEacc, SVMacc, DNNconf, CNNconf,SVMconf, TREEconf, EXPprediction, sampleName = getStats(sampleNum, accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE, predictionsEXP)

    w.itemconfig(titleText, text=sampleName)

    w.itemconfig(ENSpredictionText, text= str(ENSprediction))
    w.itemconfig(DNNpredictionText, text= str(DNNprediction))
    w.itemconfig(CNNpredictionText, text= str(CNNprediction))
    w.itemconfig(TREEpredictionText, text= str(TREEprediction))
    w.itemconfig(SVMpredictionText, text= str(SVMprediction))

    w.itemconfig(EXPpredictionText, text= str(EXPprediction))
    w.itemconfig(DNNaccText, text= str(DNNacc))
    w.itemconfig(CNNaccText, text= str(CNNacc))
    w.itemconfig(TREEaccText, text= str(TREEacc))
    w.itemconfig(SVMaccText, text= str(SVMacc))

    w.itemconfig(DNNconfText, text= str(DNNconf))
    w.itemconfig(CNNconfText, text= str(CNNconf))
    w.itemconfig(TREEconfText, text= str(TREEconf))
    w.itemconfig(SVMconfText, text= str(SVMconf))

    img1 = ImageTk.PhotoImage(Image.open("graph" + str(sampleNum) + ".png").resize((244,195)))

    if predictionsEXP[sampleNum] != predictionsENS[sampleNum]:
        w.itemconfig(correctBox, fill= "pink")  
    else:
        w.itemconfig(correctBox, fill= "#98FB98")      
    

master = Tk()

sampleNum = 1

w = Canvas(master, width=500, height=500)
w.bind("<Button-1>", callback)
w.pack()
w.create_rectangle(0,0,500,500, fill = "white")
w.create_rectangle(55,2, 445, 50, fill = "grey") 
w.create_line(0,50,500,50,width = 3) 
titleText = w.create_text(250,25, text="Sample #" + str(sampleNum))
w.create_line(0,275,500,275,width = 3)
w.create_line(250,275,250,500, width = 3)
w.create_text(125,290, text="Spectra:")
w.create_text(375,290, text="Top 10 Most Important Features:")
correctBox = w.create_rectangle(0,225, 500, 275, fill = "#98FB98")  
ENSpredictionText = w.create_text(125,250, text = ENSprediction)
EXPpredictionText = w.create_text(375,250, text = EXPprediction)
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
DNNpredictionText = w.create_text(62,100, text = DNNprediction)
DNNaccText = w.create_text(62,135, text = DNNacc)
DNNconfText = w.create_text(62,170, text = DNNconf)


w.create_text(187,65, text = "CNN:")
CNNpredictionText = w.create_text(187,100, text = CNNprediction)
CNNaccText = w.create_text(187,135, text = CNNacc)
CNNconfText = w.create_text(187,170, text = CNNconf)

w.create_text(312,65, text = "SVM:")
SVMpredictionText = w.create_text(312,100, text = SVMprediction)
SVMaccText = w.create_text(312,135, text = SVMacc)
SVMconfText = w.create_text(312,170, text = SVMconf)


w.create_text(437,65, text = "TREE:")
TREEpredictionText = w.create_text(437,100, text = TREEprediction)
TREEaccText = w.create_text(437,135, text = TREEacc)
TREEconfText = w.create_text(437,170, text = TREEconf)

from PIL import ImageTk, Image

img1 = ImageTk.PhotoImage(Image.open("graph0.png").resize((244, 195)))
img2 = ImageTk.PhotoImage(Image.open("featureChart.png").resize((247, 195)))

image1 = w.create_image(0,305,anchor = NW, image = img1)
w.create_image(256,305, anchor = NW, image = img2)

mainloop()