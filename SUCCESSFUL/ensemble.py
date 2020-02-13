import math
import time
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Flatten, Dropout, MaxPooling1D
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

def startup():
    pd.options.display.max_rows = 8
    pd.options.display.max_columns = 9
    pd.options.display.float_format = '{:.6f}'.format
    pd.set_option('mode.chained_assignment', None)

    dataset = pd.read_csv("Dataset_Github_Labeled.csv")
    #addNoise(dataset, 1) #second number for 2^power size i.e. 1->2 times, 2-> 4 times, 3->8 times, takes a while
    x= dataset.drop(['class'], axis=1)
    y= dataset['class']
    for i in range (0,dataset.shape[0]):
        if y[i].startswith('High-grade'):
            y[i] = 'High-grade'
        elif y[i].startswith('Low-grade'):
            y[i] = 'Low-grade'
        elif y[i].startswith('Normal'):
            y[i] = 'Normal'

    #split into train/test/val
    x_train, y_train, x_validation, y_validation, x_test, y_test = splitData(dataset)
    #construct feature columns
    x_labels = x.head(0)
    feature_columns=construct_feature_columns(x_labels)

    modelSVM, modelDNN, modelTREE, modelCNN, accuracyCNN, predictionsCNN = trainModels(feature_columns, x_train, y_train,x_test,y_test)
    ptemp = list(modelDNN.predict(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False)))
    predictionsDNN = []
    for i in range(len(ptemp)):
        predictionsDNN.append(ptemp[i]["probabilities"])
    predictionsSVM = modelSVM.predict_proba(x_test) 
    predictionsTREE = modelTREE.predict_proba(x_test)

    accuracySVM, accuracyDNN, accuracyTREE, accuracyCNN = getAccuracy(x_test, y_test, modelSVM, modelDNN, modelTREE, accuracyCNN)
    predictionsENS = []

    for i in range(len(predictionsDNN)):
        tempDNN = predictionsDNN[i]
        tempCNN = predictionsCNN[i]
        tempTREE = predictionsTREE[i]
        tempSVM = predictionsSVM[i]
        score0 = accuracyDNN*tempDNN[0] + accuracyCNN*tempCNN[0] + accuracyTREE*accuracyTREE*tempTREE[0] + accuracySVM*tempSVM[0]
        score1 = accuracyDNN*tempDNN[1] + accuracyCNN*tempCNN[1] + accuracyTREE*accuracyTREE*tempTREE[1] + accuracySVM*tempSVM[1]
        score2 = accuracyDNN*tempDNN[2] + accuracyCNN*tempCNN[2] + accuracyTREE*accuracyTREE*tempTREE[2] + accuracySVM*tempSVM[2]
        if score0 > score1 and score0 > score2:
            predictionsENS.append(0)
        elif score1 > score0 and score1 > score2:
            predictionsENS.append(1)
        elif score2 > score0 and score2 > score1:
            predictionsENS.append(2)

    finalAccuracy = accuracy_score(predictionsENS, y_test)
    print('DNN:' , accuracyDNN)
    print('CNN:', accuracyCNN)
    print('SVM:', accuracySVM)
    print('TREE: ', accuracyTREE)
    print('...................')
    print('ENS:' ,finalAccuracy)
    print(confusion_matrix(predictionsENS, y_test))

    return accuracyDNN, accuracyCNN, accuracySVM, accuracyTREE, predictionsENS, predictionsDNN, predictionsCNN, predictionsSVM, predictionsTREE,y_test

def addNoise(data,powerIN):
    sd = dataset.std(axis=0)
    m = dataset.mean(axis = 0)
    original = (data.shape[0])
    datacopy = data.copy()
    frames = [data,datacopy]
    for i in range(powerIN): #how many duplicates
        data2 = pd.concat(frames,ignore_index=True)
    for i in range(original, data.shape[0]):
        for j in range (1,1365):
            data.iloc[i,j]+=(np.random.normal(m[j], sd[j], 1))[0]
    return data

def splitData(dataset):
    training, validation, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))]) 
    x_train = training.drop(['class'], axis=1)
    y_train = training['class']
    x_validation=validation.drop(['class'], axis=1)
    y_validation=validation['class']
    x_test=test.drop(['class'], axis=1)
    y_test=test['class']
    lbl_encoder = LabelEncoder()
    y_train= lbl_encoder.fit_transform(y_train)
    y_test= lbl_encoder.fit_transform(y_test)
    y_validation= lbl_encoder.fit_transform(y_validation)
    return x_train, y_train, x_validation, y_validation, x_test, y_test

    # Convert numeric features into Dense Tensors, and construct the feature columns
def construct_feature_columns(input_features_DataFrame):
    tensorSet = ([])
    for elem in input_features_DataFrame:
        tensorSet.append( tf.feature_column.numeric_column(str(elem)) )
    return tensorSet

    # Create the input function for training + evaluation. boolean = True for training.
def input_fn(features, labels, training=True, batch_size=32 ):
    dataf = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataf = dataf.shuffle(200).repeat()
    return dataf.batch(batch_size=batch_size)

def trainModels(feature_columns, x_train, y_train,x_test,y_test):
    modelCNN,accuracy,predictions = trainCNN(x_train,y_train,x_test, y_test)
    modelSVM = trainSVM(x_train, y_train)
    modelDNN = trainDNN(feature_columns, x_train, y_train)
    modelTREE = trainTree(feature_columns, x_train, y_train)
    return modelSVM, modelDNN, modelTREE, modelCNN, accuracy,predictions

def trainTree(feature_columns, x_train, y_train):
    treeclassifier = tree.DecisionTreeClassifier()
    treeclassifier = treeclassifier.fit(x_train, y_train)

    return treeclassifier

def trainCNN(trainX, trainY, testX, testY):
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

    return model, accuracy, OP

def trainDNN(feature_columns, x_train, y_train):
    learning_rate=0.001
    if (tf.__version__[0] == '2'):
        optimizer_adam= tf.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    hidden_units=[37,30,19]
    model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns,  optimizer=optimizer_adam, n_classes=3)
    #model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=1000) # originally steps=1000 from template
    return model

def trainSVM(x_train, y_train):
    model = SVC(kernel='linear', probability = True)
    model.fit(x_train, y_train)
    return model

def getAccuracy(x_test, y_test, modelSVM, modelDNN, modelTREE, accuracyCNN):
    pSVM = modelSVM.predict(x_test)
    accuracySVM = accuracy_score(pSVM,y_test)
    pTREE = modelTREE.predict(x_test)
    accuracyTREE = accuracy_score(pTREE, y_test)
    pDNN = modelDNN.evaluate(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False), steps=1)
    accuracyDNN = pDNN["accuracy"]

    return accuracySVM, accuracyDNN, accuracyTREE, accuracyCNN



