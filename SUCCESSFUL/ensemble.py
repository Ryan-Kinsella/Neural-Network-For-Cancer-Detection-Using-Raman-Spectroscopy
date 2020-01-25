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
        y[i] = 'Cancer'
    elif y[i].startswith('Low-grade'):
        y[i] = 'Cancer'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'

def addNoise(data,powerIN):
    sd = dataset.std(axis=0)
    m = dataset.mean(axis = 0)
    original = (data.shape[0])
    datacopy = data.copy()
    frames = [data,datacopy]
    for i in range(powerIN): #how many duplicates
        data2 = pd.concat(frames,ignore_index=True)
    for i in range(original, data.shape[0]):
        print(i)
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

def trainModels(feature_columns, x_train, y_train):
    print('STARTING TO TRAIN THE MODELS--------------------------------------------------')
    modelSVM = trainSVM(x_train, y_train)
    print('FINISHED THE SVM--------------------------------------------------------------')
    modelDNN = trainDNN(feature_columns, x_train, y_train)
    print('FINISHED THE DNN--------------------------------------------------------------')
    modelTREE = trainTree(feature_columns, x_train, y_train)
    print('FINISHED THE TREE-------------------------------------------------------------')

    return modelSVM, modelDNN, modelTREE

def trainTree(feature_columns, x_train, y_train):
    model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer = 1)
    model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), max_steps=1000)

    return model

def trainDNN(feature_columns, x_train, y_train):
    learning_rate=0.001
    optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    hidden_units=[37,30,19]
    model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns,  optimizer=optimizer_adam, n_classes=2)
    model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=1000) # originally steps=1000 from template
    return model

def trainSVM(x_train, y_train):
    model = SVC(kernel='linear', probability = True)
    model.fit(x_train, y_train)
    return model

#split into train/test/val
x_train, y_train, x_validation, y_validation, x_test, y_test = splitData(dataset)
#construct feature columns
x_labels = x.head(0)
feature_columns=construct_feature_columns(x_labels)

modelSVM, modelDNN, modelTREE = trainModels(feature_columns, x_train, y_train)
ptemp = list(modelDNN.predict(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False)))
predictionsDNN = ptemp["probabilities"]
ptemp =list(modelTREE.predict(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False)))
predictionsTREE= ptemp["probabilities"]
predictionsSVM = modelSVM.predict_proba(x_test)

print (predictionsDNN[0])
print (predictionsTREE[0])
print (predictionsSVM[0])

