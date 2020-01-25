"""
Neural network for the detection of cancerous tissues using raman spectroscopy.
Using data set "Dataset_Github_csv.csv", taken from paper
H Chen, X Li, N Broderick ,Y Liu, Y Zhou, W Xu. Identification and characterization of bladdercancer by low-resolution fiber-optic Raman spectros-copy. J. Biophotonics. 2018;e201800016.https://doi.org/10.1002/jbio.201800016.
https://github.com/haochen23/bladderMachineLearning

Other Sources:
Source 1: Intro to neural nets, https://colab.research.google.com/notebooks/mlcc/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=introneuralnets-colab&hl=en#scrollTo=RWq0xecNKNeG
Source 2: Improving neural net performance, https://colab.research.google.com/notebooks/mlcc/improving_neural_net_performance.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=improvingneuralnets-colab&hl=en
Source 3: pandas.DataFrame documentation, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
Source 4: Tensorflow Feature Columns, https://www.tensorflow.org/tutorials/structured_data/feature_columns
Source 5: Tensorflow DNNClassifier, https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier
Source 6: https://towardsdatascience.com/getting-data-into-tensorflow-estimator-models-3432f404a8da
Source 7: Someone can use sns.countplot and sns.heatmap as shown in this article to see correlations,
    https://medium.com/datadriveninvestor/tensorflow-dnnclassifier-4e68df3df00
Source 8: Similar to Source 7, How to use a tensorflow pipeline for premade estimators, https://www.tensorflow.org/tutorials/estimator/premade


Notes:
Terminology:
    Targets: cancer vs not cancer, the last feature column with High-grade tumor, Low-grade
        tumor, and Normal as the three classes.
    Features: what's used in predicting the targets, ie. the 1-to-class columns of the Dataset_Github_Labeled.csv file.

Ryan - I am using Dataset_Github_Labeled.csv
     - I mainly used Source 7 as a template for the code in Test.py.
    - Using The Keras API is apparently easier to learn.

To do:
    A non-random forest
    Using the outputs of each separate model, use the majority of what the classifiers have predicted to classify.
    pseudocode,
    for row in dataFrame:
        final_model[row] = most_occuring_integer_from_all_models (NN_model[row] = either 0,1, or 2 , Bayes_model[row], SVM_model[row], etc...)
        #so the final_model will have the majority of what each model predicts for each, which in theory should limit the outliers of each class predictor


    Someone needs to figure out using seaborn to use with a pandas dataframe, Source 7.

"""
################## Source 7 Mainly used, Source 8 helps explain as well ##################
# Source 1 can be used to understand pre-processing the features and targets

import math # unused
import time
from IPython import display # unused
import matplotlib.pyplot as plt # unused
import numpy as np # unused
import pandas as pd
from sklearn import metrics # unused
import tensorflow as tf
import seaborn as sns # unused

def addNoise(data):
    sd = dataset.std(axis=0)
    m = dataset.mean(axis = 0)
    original = (data.shape[0])
    datacopy = data.copy()
    frames = [data,datacopy]
    for i in range(1): #how many duplicates
        data2 = pd.concat(frames,ignore_index=True)
    print('------------------------------------')
    print(data2)
    print(data2.shape[0])
    print('------------------------------------')
    for i in range(original, data.shape[0]):
        print(i)
        for j in range (1,1365):
            data.iloc[i,j]+=(np.random.normal(m[j], sd[j], 1))[0]
    return data

pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('mode.chained_assignment', None)
dataset = pd.read_csv("Dataset_Github_Labeled.csv")
#dataset = addNoise(dataset) #ADDS UP TO 2500 ROWS
# From DataFrame, split into features (x) and labels (y)
x= dataset.drop(['class'], axis=1)
y= dataset['class']
# change y in the csv file to be assigned to one of three classes: High-grade, Low-grade, Normal
for i in range (0,dataset.shape[0]): # 0 - 323, same size as x
    #print(type(y[i]))
    if y[i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        y[i] = 'High-grade'
    elif y[i].startswith    ('Low-grade'):
        y[i] = 'Low-grade'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'


# Encode target variable (y)
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y= lbl_encoder.fit_transform(y)
# print(y) # shows how the classes are numerically assigned through this change, by 0,1, or 2

# Split data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25) # 75% training, 25% test

# Convert numeric features into Dense Tensors, and construct the feature columns
def construct_feature_columns(input_features_DataFrame):
  tensorSet = ([])
  for elem in input_features_DataFrame:
    tensorSet.append( tf.feature_column.numeric_column(str(elem)) ) # where elem is a str feature label
  return tensorSet
x_labels = x.head(0) # gets the labels for x, with the dropped class column
feature_columns=construct_feature_columns(x_labels)
print(x_labels)


# Create the input function for training + evaluation. boolean = True for training.
def input_fn(features, labels, training=True, batch_size=32 ):
    dataf = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataf = dataf.shuffle(200).repeat() # shuffle ~<half the data
    return dataf.batch(batch_size=batch_size)

# Think about adding additional classifiers and comparing their accuracies (like a random forest). 
timer_start = time.time()
learning_rate=0.001
print("Build optimizer: Learning rate = " , learning_rate)
optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
hidden_units=[37,30,19]
print("Building model: Hidden units = " , hidden_units)
model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns,  optimizer=optimizer_adam, n_classes=3)
print("Training model...")
model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=1000) # originally steps=1000 from template
print("Evaluate model...")
predictions = list(model.predict(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False)))
#THIS GIVES AN ARRAY OF CLASS PROBABILITIES - use this w/ model accuracy or whatever
print(predictions[0]["probabilities"])
eval_results = model.evaluate(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False), steps=1)
end_timer = time.time()
print("Model training and testing took ", round(end_timer - timer_start, 1), " seconds.")
print(eval_results)
