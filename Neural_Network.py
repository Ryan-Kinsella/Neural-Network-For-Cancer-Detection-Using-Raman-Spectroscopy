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
     - I mainly used Source 1 as a template for this code. 





"""




###############################################################################################################
# Setup
###############################################################################################################
import math
from IPython import display # unused
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics # unused
import tensorflow as tf
import seaborn as sns # unused

from tensorflow.python.data import Dataset
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format

###############################################################################################################
# Functions
###############################################################################################################

# We can add to if we want to multiply certain features by 10^x, or other manipulations to make data
# easier to understand, instead of changing the csv file.
def preprocess_features(dataframe):
    """Prepares input features from data set.

    Args:
      dataframe: A Pandas DataFrame expected to contain data
        from the data set.
    Returns:
      A DataFrame that contains ONLY the features to be used for the model, including
      synthetic features if needed.
    """
    processed_features = dataframe.copy()
    processed_features = processed_features.drop(labels='High-grade tumor-1', axis=1) # Remove targets
    return processed_features

# If we decide we only want to select a certain number of features, can do so here instead of
# changing the csv file.
def preprocess_targets(dataframe):
    """Prepares target features (i.e., labels) from data set.

    Args:
      dataframe: A Pandas DataFrame expected to contain data
        from the data set.
    Returns:
      A DataFrame that contains ONLY the target feature.
    """
    processed_targets = dataframe.pop('High-grade tumor-1')  # Remove targets
    return processed_targets


# # ALTERNATIVE TO THE TWO FUNCTIONS ABOVE
# # (Source 4) We will wrap the dataframes with tf.data. This will enable us to use feature columns as a bridge to map
# # from the columns in the Pandas dataframe to features used to train the model.
# # A utility method to create a tf.data dataset from a Pandas Dataframe
# def df_to_dataset(dataframe, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy() # keep original contents of dataframe
#     labels = dataframe.pop('High-grade tumor-1') # Creates a 1D dataframe for the labels, ie. what we're classifying
#     ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
#     if shuffle:
#         ds = ds.shuffle(len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])


# # Source 1 my_input function, for clarification for how it works see Source 5. Returns a tuple of
# # Input builders
# def input_fn_train_and_eval(features, targets, batch_size=1, shuffle=True, num_epochs=None):
#     """Trains a neural net regression model.
#
#     Args:
#       features: pandas DataFrame of features
#       targets: pandas DataFrame of targets
#       batch_size: Size of batches to be passed to the model
#       shuffle: True or False. Whether to shuffle the data.
#       num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
#     Returns:
#       Tuple of (features, labels) for next data batch
#     """
#     # Convert pandas data into a dict of np arrays.
#     features = {key: np.array(value) for key, value in dict(features).items()}
#
#     # Construct a dataset, and configure batching/repeating.
#     ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
#     ds = ds.batch(batch_size).repeat(num_epochs)
#
#     # Shuffle the data, if specified.
#     if shuffle:
#         ds = ds.shuffle(10000)
#
#     # Return the next batch of data.
#     features, labels = ds.make_one_shot_iterator().get_next()
#     return features, labels

# Required for estimator.train(). Seperates features and targets.
"""
x = training_examples[[feature_column_list]]
y = training_examples[label_column_name]
"""
def input_fn_train (dataframe, name_target):
    # Returns tf.data. Dataset of (x, y) tuple where y represents label's class
    # index.

    return tf.estimator.inputs.pandas_input_fn(
      x=dataframe,
      y=dataframe[name_target],
      batch_size = 32,
      num_epochs = 5,
      shuffle = True,
      queue_capacity = 2000,
      num_threads = 1
    )



# def input_fn_predict:
#   # Returns tf.data.Dataset of (x, None) tuple.
#   pass

###############################################################################################################
# Main
###############################################################################################################

# Source 1,
dataframe = pd.read_csv("Dataset_Github_csv.csv")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
print ("DataFrame read.")
print ("DataFrame contents: [323 rows x 1368 columns]")
# print (dataframe)


# Choose the first 75%, 243 (out of 324) rows for training.
training_features = preprocess_features(dataframe.head(243))
training_targets = preprocess_targets(dataframe.head(243))
training_dataframe = dataframe.head(243)


# Choose the last 25%, 81 (out of 324) rows for validation.
validation_features = preprocess_features(dataframe.tail(81))
validation_targets = preprocess_targets(dataframe.tail(81))

# # Output dataframe subsets.
# print("Training features summary:")
# display.display(training_features.describe())
# print("Validation features summary:")
# display.display(validation_features.describe())
#
# print("Training targets summary:")
# display.display(training_targets.describe())
# print("Validation targets summary:")
# display.display(validation_targets.describe())

# Source 5,
# Using the ProximalAdagradOptimizer optimizer with regularization. Can play around with diff options.
model = tf.compat.v2.estimator.DNNClassifier(
    feature_columns=construct_feature_columns(training_features),
    # Using rule of thumb for hidden layers, square root of the features. We should play around with the testing of this.
    hidden_units=[37, 6], # The format is hidden_units = [numberOfNodesInFirstLayer, numberOfNodesInSecondLayer, etc.]
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Source 5,
# Format in which the estimator evaluates the performance of the model.
batch_size = 5
#estimator.train(input_fn=input_fn_train_and_eval(features = training_features, targets= training_targets,batch_size=batch_size,shuffle=True,num_epochs=None))
model.train(input_fn=lambda : input_fn_train(training_features, 'High-grade tumor-1')) # High-grade tumor-1 is the string label for class
#metrics = estimator.evaluate(input_fn=input_fn_train_and_eval(         ))
#predictions = estimator.predict(input_fn=input_fn_predict)



"""
  Removed the binary T/F indicators for classification in the csv file. 
  For isolating classification information at the end of the DataFrame, you can check using the following:
  if "High_grade" in dataframe.tail(1): #if the last column contains text "High-grade", etc below. 
  if "Low-grade" in dataframe.tail(1):
  if "Normal" in dataframe.tail(1):
"""