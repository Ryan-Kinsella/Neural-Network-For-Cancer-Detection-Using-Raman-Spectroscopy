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
        y[i] = 'High-grade'
    elif y[i].startswith('Low-grade'):
        y[i] = 'Low-grade'
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
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: DataFrame with the names of the numerical input features to use.
  Returns:
    A set of tf numeric feature columns
  """
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
timer_start = time.time()
learning_rate=0.001
print("Build optimizer: Learning rate = " , learning_rate)
optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
hidden_units=[37,30,19]
print("Building model: Hidden units = " , hidden_units)
model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns,  optimizer=optimizer_adam, n_classes=3)
print("Training model...")
model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=1000) # originally steps=1000 from template
print("Validate model...")
validation_results = model.evaluate(input_fn=lambda: input_fn(features=x_validation, labels=y_validation, training=False), steps=1)
print("Testing model...")
testing_results = model.evaluate(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False), steps=1)

end_timer = time.time()
print("Model training and testing took ", round(end_timer - timer_start, 1), " seconds.")
print("Validation results: ")
print(validation_results)
print("Testing results: ")
print(testing_results)
print("hidden_units=",hidden_units,"                                ")

"""
############# CHOOSING HIDDEN UNITS #############
"Artificial Intelligence for Humans, Volume 3: Deep Learning and Neural Networks" ISBN: 1505714346
Traditionally, neural networks only have three types of layers: hidden, input and output. 

General rule of thumbs: (all different)
A: The number of hidden neurons should be between the size of the input layer and the size of the output layer.
B: The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
C: The number of hidden neurons should be less than twice the size of the input layer.

The number of layers are described below. We will be using 3 layers total since we don't want to be overfitting the 243 training cases.  
1   :	Can approximate any function that contains a continuous mapping from one finite space to another.
2   :	Can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy.
>2	:   Additional layers can learn complex representations (sort of automatic feature engineering) for layer layers.

####################### Testing Model Results: Using steps=1000, learning rate=0.001, features=1367 #######################
Random inputs:
hidden_units=[37,37,37]                                            = 85.19% accuracy
hidden_units=[37,18,6 ]                                            = 88.89% accuracy
hidden_units=[50,30,19]                                            = 88.89% accuracy
hidden_units=[37,25,17]                                            = 90.12% accuracy
hidden_units=[37,28,17]                                            = 91.35% accuracy 
hidden_units=[37,30,19]                                            = 93.82% accuracy

A: The number of hidden neurons should be between the size of the input layer and the size of the output layer.
1367 input features, hidden_units=[30,37,19]                       = 92.59% accuracy

B: The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
hidden_units=[31,37,6]                                             = 95.06% accuracy
hidden_units=[33,37,8]                                             = 91.35% accuracy
hidden_units=[32,37,7]                                             = 83.35% accuracy
hidden_units=[30,37,5]                                             = 93.38% accuracy
hidden_units=[29,37,4]                                             = 93.38% accuracy
hidden_units=[28,37,3]                                             = 93.38% accuracy
hidden_units=[34,37,9]                                             = 92.59% accuracy
hidden_units=[37,37,12]                                            = 93.83% accuracy



C: The number of hidden neurons should be less than twice the size of the input layer.
hidden_units=[14,37,19]                                            = 93.83% accuracy
hidden_units=[13,37,19]                                            = 91.58% accuracy
hidden_units=[17,37,19]                                            = 87.58% accuracy

DNN: more than 3 layers
hidden_units=[14,37,19, 6]                                         = 91.36% accuracy



####################### Testing Model Results: Using steps=1000, learning rate=0.001, features=1367 #######################
"""


"""
Nov. 27 notes:

- add validation set, compare results to the test set.
- essentially try our best to automate everything so it will perform well on a new data set,
  providing explanations on why we chose different methods. 
- consider using sequential feature selection, ie. choose a (correlated?) feature, then add another one and see if it adds 
- batch size ~32, test with it
- maybe scrap the sklearn top 20% data set in place of sequential feature selection
  OR try with top 5-10% since our data set is short and wide
- For NN, the rule of thumb is for number of SAMPLES, not the input layer



"""