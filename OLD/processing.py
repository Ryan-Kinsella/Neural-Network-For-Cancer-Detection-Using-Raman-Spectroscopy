#cmd
#    C:\python27\python.exe processing.py
# cd C:\Users\mikem\Documents\Queen's\Senior\498\Neural-Network-For-Cancer-Detection-Using-Raman-Spectroscopy\mike\

#mike - oct 6/19


#Data processing - converting the raw csv into an array
#this uses the github dataset

import csv, random
import matplotlib.pyplot as plt

vals = []

def importData():
    size = 1368
    #1367 is the label
    #numerical data ends at 1366
    #not bringing in booleans rn
    global vals
    with open('data.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            curr = [0 for i in range (size)]
            for col in range (size):
                curr[col] = row[col]
            vals.append(curr)
    #this is important for formatting
    vals[0][0] = vals[0][0][3:]
    #have to get in floats cause really small numbers break it
    for i in range(0, len(vals)):
        for j in range(0, len(vals[0])-1):
            if float(vals[i][j])< 0.01:
                vals[i][j] = float(vals[i][j])
    #process
    for i in range (len(vals)):
        temp = vals[i][1367]
        valid = temp.find('Normal')
        #if normal is not found, set label to -1 , it has tumour
        if valid == -1:
            vals[i][1367] = -1
        #if normal is found, set label to 1, it doesn't have tumour
        else:
            vals[i][1367] = 1


def draw():
    #this plots stuff. big spike around 1k
    global vals
    temparray = []
    for j in range(100,101):    
        for i in range(0,1366):
            temparray.append(float(vals[j][i]))
            plt.plot(temparray)
    
    plt.show()

#mainline
print ('run')
importData()
draw()
print ('done')