import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from string import digits

#bankdata = pd.read_csv("C:/Users/saman/PycharmProjects/SVMtester/bill_authentication.csv")
cancerdata = pd.read_csv("Dataset_Github_Labeled.csv")


cancerdata.shape
cancerdata.head()

#edit so it says High, Low, Normal
#remove "-grade tumor-***" or "-***"
count=0
for row in cancerdata[cancerdata.columns[-1]]:
    #print(row)
    if "-grade tumor-" in row:
        #print('"-grade tumor-***" string removed')
        remove_digits = row.maketrans('', '', digits)
        row = row.translate(remove_digits)
        row = row.replace('-grade tumor-', '')

    if "Normal-" in row:
        #print('"-***" string removed')
        remove_digits = row.maketrans('', '', digits)
        row = row.translate(remove_digits)
        row = row.replace('-', '')
    #print(row)
    cancerdata[cancerdata.columns[-1]][count]=row
    count=count+1

print(cancerdata[cancerdata.columns[-1]])


#data preprocessing
values=['High','Low','Normal']
X = cancerdata.drop(cancerdata.columns[-1], axis=1)
y = cancerdata[cancerdata.columns[-1]]


#Once the data is divided into attributes and labels, the final preprocessing step is to divide data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


#train
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear', probability = True)
svclassifier.fit(X_train, y_train)

#predict
y_pred = svclassifier.predict(X_test)
class_probabilities = svclassifier.predict_proba(X_test)
print(class_probabilities)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
#we don't need this i don't think
#print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))