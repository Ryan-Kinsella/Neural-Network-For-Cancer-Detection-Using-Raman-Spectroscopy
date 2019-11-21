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
        final_model[row] = most_occuring_integer_from_all_models (NN_model[row] = the class is either 0,1, or 2 , Bayes_model[row], SVM_model[row], etc...)
        #so the final_model will have the majority of what each model predicts for each, which in theory should limit the outliers of each class predictor


    Someone needs to figure out using seaborn to use with a pandas dataframe, Source 7.

TA: Abeshek* meetings

"""
################## Source 7 Mainly used, Source 8 helps explain as well ##################
# Source 1 can be used to understand pre-processing the features and targets

import math # unused
import time
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


"""
# This was taken and used in select_best_twenty_percent_features.ipynb
dataset = pd.read_csv("Dataset_Github_Labeled.csv")
# From DataFrame, split into features (x) and labels (y)
x= dataset.drop(['class'], axis=1)
y= dataset['class']
# change y in the csv file to be assigned to one of three classes: High-grade, Low-grade, Normal
for i in range (0,324): # 0 - 323, for all records not including labels
    #print(type(y[i]))
    if y[i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        y[i] = 'High-grade'
    elif y[i].startswith('Low-grade'):
        y[i] = 'Low-grade'
    elif y[i].startswith('Normal'):
        y[i] = 'Normal'
# GIVES WARNINGS for slicing although it still works, see print below
# Warnings can be turned back on by deleting pd.set_option('mode.chained_assignment', None)
# print (y)
"""

dataset = pd.read_csv("useful_features.csv") # 324 rows, the top 20% 275 feature columns
x= dataset.drop(['class'], axis=1)
y= dataset['class']

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
print(x_labels)
# print(type(feature_columns))


# Create the input function for training + evaluation. boolean = True for training.
def input_fn(features, labels, training=True, batch_size=100 ):
    dataf = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataf = dataf.shuffle(1000).repeat()
    return dataf.batch(batch_size=batch_size)

# Think about adding additional classifiers and comparing their accuracies (like a random forest).
timer_start = time.time()
learning_rate=0.001
print("Build optimizer: Learning rate = " , learning_rate)
optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
hidden_units=[60,58,12]
print("Building model: Hidden units = " , hidden_units)
model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns,  optimizer=optimizer_adam, n_classes=3)
print("Training model...")
model.train(input_fn=lambda: input_fn(features=x_train, labels=y_train, training=True), steps=1000) # originally steps=1000 from template
print("Evaluate model...")
eval_results = model.evaluate(input_fn=lambda: input_fn(features=x_test, labels=y_test, training=False), steps=1)
# predictions_DNN = model.predict(input_fn=lambda  : input_fn(features=x_test, labels=y_test, training=False, batch_size=1))
end_timer = time.time()
print("Model training and testing took", round((end_timer - timer_start), 2), "seconds. ")
print("hidden_units=",hidden_units,"                                ")
print(eval_results)
# print(predictions_DNN)


"""
############# CHOOSING HIDDEN UNITS #############
"Artificial Intelligence for Humans, Volume 3: Deep Learning and Neural Networks" ISBN: 1505714346
Traditionally, neural networks only have three types of layers: hidden, input and output. 
https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
http://hagan.okstate.edu/NNDesign.pdf#page=469

Max_num_hidden_neurons = samples_training / ( alpha * ( input_neurons + output_neurons) )
, where alpha is an arbitrary scaling factor between 2-10. Don't know input neurons

One issue within this subject on which there is a consensus is the performance difference from adding additional hidden layers: the situations in which performance
improves with a second (or third, etc.) hidden layer are very few. One hidden layer is sufficient for the large majority of problems.

General rule of thumbs: (all different)
A: The number of hidden neurons should be between the size of the input layer and the size of the output layer.
B: The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
C: The number of hidden neurons should be less than twice the size of the input layer.

The number of layers are described below. We will be using 3 layers total since we don't want to be overfitting the 243 training cases.  
1   :	Can approximate any function that contains a continuous mapping from one finite space to another.
2   :	Can represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy.
>2	:   Additional layers can learn complex representations (sort of automatic feature engineering) for layer layers.
"""

"""
####################### Testing Model Results for useful_features.csv: Using steps=1000, learning rate=0.001, features=275 #######################

A: Will always be true.
B: The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
= 2/3 * 275 + 3
= 186 hidden neurons total.
C: The number of hidden neurons should be less than twice the size of the input layer.
> 1/2 * 275
> 137 neurons total
Therefore, use 137 < hidden neurons < 186


For adding hidden layers, using sqrt(275) = 16. 
hidden_units= [16, 16, 16]                                 
{'accuracy': 0.8271605, 'average_loss': 0.8205901, 'loss': 66.4678, 'global_step': 1000}
hidden_units= [16, 16, 16, 16]                                 
{'accuracy': 0.7777778, 'average_loss': 1.2065957, 'loss': 97.734245, 'global_step': 1000}
hidden_units= [16, 16, 16, 16, 16]                                 
{'accuracy': 0.80246913, 'average_loss': 0.99092764, 'loss': 80.26514, 'global_step': 1000}
hidden_units= [16, 16, 16, 16, 16, 16]                                 
{'accuracy': 0.8148148, 'average_loss': 1.2898284, 'loss': 104.476105, 'global_step': 1000}
hidden_units= [16, 16, 16, 16, 16, 16, 16]                                 
{'accuracy': 0.7777778, 'average_loss': 1.5623556, 'loss': 126.550804, 'global_step': 1000}
Will use 3 hidden layers. 

For 137 < hidden neurons < 186:
hidden_units= [60, 60, 40]                                 
{'accuracy': 0.8518519, 'average_loss': 0.60328394, 'loss': 48.866, 'global_step': 1000}
hidden_units= [50, 50, 50]                                 
{'accuracy': 0.83950615, 'average_loss': 0.8168741, 'loss': 66.1668, 'global_step': 1000}
hidden_units= [60, 60, 60]                                 
{'accuracy': 0.80246913, 'average_loss': 0.9265667, 'loss': 75.0519, 'global_step': 1000}

For hidden neurons <= 137:
hidden_units= [32, 48, 16]                                 
{'accuracy': 0.80246913, 'average_loss': 0.46693057, 'loss': 37.821377, 'global_step': 1000}
hidden_units= [40, 40, 40]
{'accuracy': 0.8765432, 'average_loss': 0.63947505, 'loss': 51.797478, 'global_step': 1000} 
hidden_units= [42, 42, 42]                                 
{'accuracy': 0.79012346, 'average_loss': 0.865355, 'loss': 70.09376, 'global_step': 1000}
hidden_units= [40, 88, 9]                                 
{'accuracy': 0.7654321, 'average_loss': 1.1175127, 'loss': 90.518524, 'global_step': 1000}
hidden_units= [40, 40, 9]                                 
{'accuracy': 0.75308645, 'average_loss': 1.0958208, 'loss': 88.76148, 'global_step': 1000}
hidden_units= [60, 60, 17]                                 
{'accuracy': 0.83950615, 'average_loss': 1.1248006, 'loss': 91.10885, 'global_step': 1000}
hidden_units= [60, 60, 15]                                 
{'accuracy': 0.8518519, 'average_loss': 0.7505407, 'loss': 60.793793, 'global_step': 1000}
hidden_units= [60, 60, 12]                                 
{'accuracy': 0.8765432, 'average_loss': 0.5921378, 'loss': 47.96316, 'global_step': 1000}
hidden_units= [60, 60, 9]                                 
{'accuracy': 0.83950615, 'average_loss': 0.53629625, 'loss': 43.439995, 'global_step': 1000}
hidden_units= [60, 60, 11]                                 
{'accuracy': 0.7654321, 'average_loss': 1.288694, 'loss': 104.38422, 'global_step': 1000}
hidden_units= [60, 62, 12]                                 
{'accuracy': 0.79012346, 'average_loss': 0.9901336, 'loss': 80.20082, 'global_step': 1000}





####################### Testing Model Results for useful_features.csv: Using steps=1000, learning rate=0.001, features=275 #######################
"""
