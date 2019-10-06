#cmd
#           C:\python27\python.exe C:\Users\mikem\Documents\Queen's\Senior\498\Neural-Network-For-Cancer-Detection-Using-Raman-Spectroscopy\mike\processing.py


#mike - oct 6/19


#Data processing - converting the raw csv into an array
#this uses the github dataset

import csv, random

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
    #scramble so same stuff isn't all together
    random.shuffle(vals)
def printData():
    global vals
    #printing label
    for i in range(len(vals)):
        print vals[i][1367]


#mainline
print ('run')
importData()
printData()
print ('done')