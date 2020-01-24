#built of neural network. takes a long time to run, low accuracies, but works

import math # unused
import time
from IPython import display # unused
import matplotlib.pyplot as plt # unused
import numpy as np # unused
import pandas as pd
from sklearn import metrics # unused
import tensorflow as tf
import seaborn as sns # unused

pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('mode.chained_assignment', None)

dataset = pd.read_csv("Dataset_Github_Labeled.csv")
# From DataFrame, split into features (x) and labels (y)
x= dataset.drop(['class'], axis=1)
y= dataset['class']
# change y in the csv file to be assigned to one of three classes: High-grade, Low-grade, Normal
for i in range (0,324): # 0 - 323, same size as x
    #print(type(y[i]))
    if y[i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        y[i] = 'Cancer'
    elif y[i].startswith('Low-grade'):
        y[i] = 'Cancer'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'
# GIVES WARNINGS although it still works, see print below
# Warnings can be turned back on by deleting pd.set_option('mode.chained_assignment', None)
# print (y)



# Split data into train and test set
from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20) # 60% training, 20% test, 20% validation

# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
# x represents attributes, y represents class label
training, validation, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))]) # 60% test, 20% validation, 20% test split.
x_train = training.drop(['class'], axis=1)
y_train = training['class']
x_validation=validation.drop(['class'], axis=1)
y_validation=validation['class']
x_test=test.drop(['class'], axis=1)
y_test=test['class']
# Encode class label y
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y_train= lbl_encoder.fit_transform(y_train)
y_test= lbl_encoder.fit_transform(y_test)
y_validation= lbl_encoder.fit_transform(y_validation)
# print(y) # shows how the classes are numerically assigned through this change, by 0,1, or 2
del training, validation, test # clear memory

# Convert numeric features into Dense Tensors, and construct the feature columns
def construct_feature_columns(input_features_DataFrame):
  tensorSet = ([])
  for elem in input_features_DataFrame:
    tensorSet.append( tf.feature_column.numeric_column(str(elem)) ) # where elem is a str feature label
  return tensorSet
  # return set([tf.feature_column.numeric_column(my_feature)
  #             for my_feature in input_features_DataFrame])

x_labels = x.head(0) # gets the labels for x, with the dropped class column
feature_columns=construct_feature_columns(x_labels)
# print(x_labels)
# print(type(feature_columns))


# Create the input function for training + evaluation. boolean = True for training.
def input_fn(features, labels, training=True, batch_size=32 ):
    dataf = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataf = dataf.shuffle(200).repeat() # shuffle ~<half the data
    return dataf.batch(batch_size=batch_size)

# Think about adding additional classifiers and comparing their accuracies (like a random forest). 

model = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer = 1)
model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), max_steps=1000)
validation_results = model.evaluate(input_fn=lambda: input_fn(features=x_validation, labels=y_validation, training=False), steps=1)
testing_results = model.evaluate(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False), steps=1)

print (validation_results)
print (testing_results)
