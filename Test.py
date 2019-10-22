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



Notes:
Terminology:
    Targets: cancer vs not cancer, the last feature column with High-grade tumor, Low-grade
        tumor, and Normal as the three classes.
    Features: what's used in predicting the targets, ie. the 0:1366 columns of the csv file.


Ryan - The Github csv file does not contain an initial row creating the labels for the columns. Eg. there's no row
    in the top right corner saying "Cancer Type", instead they're currently labeled by "High-grade tumor-1",
    which is fine but something to refer to when referencing the last column.
     - I mainly used Source 7 as a template for the code in Test.py.





"""
################## Source 7 Mainly used ##################
# Source 1 can be used to understand pre-processing the features and targets

import math
from IPython import display # unused
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics # unused
import tensorflow as tf
import seaborn as sns # unused


pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format
# display.display(validation_features.describe())

dataset = pd.read_csv("Dataset_Github_csv.csv")
# From DataFrame, split into features (x) and labels (y)
x= dataset.drop(['High-grade tumor-1'], axis=1)
y= dataset['High-grade tumor-1']
# change y to be assigned to one of three classes: High-grade, Low-grade, Normal
for i in range (0,323): # 0 - 322, same size as x
    #print(type(y[i]))
    if y[i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        y[i] = 'High-grade'
    elif y[i].startswith('Low-grade'):
        y[i] = 'Low-grade'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'
# GIVES WARNINGS although it still works, see print below
# print (y)


#print(dataset.describe(include='all'))
#print(dataset.columns)
########### Find way to display like in source 7
#dataset_seaborn = sns.load_dataset(name='https://github.com/Ryan-Kinsella/Neural-Network-For-Cancer-Detection-Using-Raman-Spectroscopy/blob/master/Dataset_Github_csv.csv')
#sns.countplot(data=dataset_seaborn, x='0.136482218') #this only works in python running notebooks, Source 7
#plt.show()


# Encode target variable (y)
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y= lbl_encoder.fit_transform(y)
# print(y) # shows how the classes are numerically assigned through this change

# Split data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.25)

# Convert numeric features into Dense Tensors, and construct the feature columns
def construct_feature_columns(input_features_DataFrame):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: DataFrame with the names of the numerical input features to use.
  Returns:
    A set of tf numeric feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features_DataFrame])
# def construct_feature_columns_2(x_labels):
#   """Construct the TensorFlow Feature Columns.
#
#   Args:
#     input_features: DataFrame with the names of the numerical input features to use.
#   Returns:
#     A set of tf numeric feature columns
#   """
#   aSet = set ([])
#   for i in range(0,322):
#       aSet.add(x_labels[0][i])
#   return aSet

x_labels = x.head(0) # gets the labels for x, with the dropped class column
# x_labels_string = x_labels.to_string()
# print(x_labels_string)
# print("Type is:" , type(x_labels_string))
# for col in x_labels_string:
#     print (col)
#x_labels_string.unique()
#print (x_labels_string)
# aSet = set (x_labels_string)
# print("X labels: ", x_labels)
# feature_col = construct_feature_columns_2(x_labels)
# print("Feature Cols: ", feature_col)
# print ("X_labels: ", x_labels_string)
# print ("aSet", aSet)


# Create the input function for training + evaluation
def train_input_fn(x_training, y_training):
    dataf = tf.data.Dataset.from_tensor_slices((dict(x_training), y_training))
    dataf = dataf.shuffle(1000).repeat().batch(10)
    return dataf
def eval_input_fn(x_testing, y_testing):
    dataf = tf.data.Dataset.from_tensor_slices((dict(x_testing), y_testing))
    return dataf.shuffle(1000).repeat().batch(10)

optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
# NEED TO SET PROPER FEATURE_COLUMNS
model=tf.estimator.DNNClassifier(hidden_units=[9], feature_columns=x_labels,  optimizer=optimizer_adam, n_classes=3)
model.train(input_fn=lambda: train_input_fn(x_train, y_train), steps=100) # originally steps=1000 from template

eval_results = model.evaluate(input_fn=eval_input_fn(x_test, y_test), steps=1)
#print(eval_results)

