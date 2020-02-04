import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Dropout, MaxPooling1D

dataset = pd.read_csv("Dataset_Github_Labeled.csv")

x= dataset.drop(['class'], axis=1)
y= dataset['class']

for i in range (0,dataset.shape[0]): # 0 - 323, same size as x
    #print(type(y[i]))
    if y[i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        y[i] = 'High-grade'
    elif y[i].startswith    ('Low-grade'):
        y[i] = 'Low-grade'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'


from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y= lbl_encoder.fit_transform(y)

from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(x,y, test_size=0.25)

trainY = trainY - 1
testY = testY - 1

trainY = keras.utils.to_categorical(trainY)
testY = keras.utils.to_categorical(testY)

verbose, epochs, batch_size = 0, 10, 32
trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)

n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
_, accuracy = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
OP = model.predict_proba(testX)

print(accuracy)
print(OP)