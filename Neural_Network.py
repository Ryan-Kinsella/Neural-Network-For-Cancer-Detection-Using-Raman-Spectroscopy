"""
Neural network for the detection of cancerous tissues using raman spectroscopy.
Using data set "Dataset_Github_csv.csv", taken from paper
H Chen, X Li, N Broderick ,Y Liu, Y Zhou, W Xu. Identification and characterization of bladdercancer by low-resolution fiber-optic Raman spectros-copy. J. Biophotonics. 2018;e201800016.https://doi.org/10.1002/jbio.201800016.
https://github.com/haochen23/bladderMachineLearning

Other Sources:
Source 1: Intro to neural nets, https://colab.research.google.com/notebooks/mlcc/intro_to_neural_nets.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=introneuralnets-colab&hl=en#scrollTo=RWq0xecNKNeG
Source 2: Improving neural net performance, https://colab.research.google.com/notebooks/mlcc/improving_neural_net_performance.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=improvingneuralnets-colab&hl=en
Source 3: pandas.DataFrame documentation, https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

"""

###############################################################################################################
# Setup
###############################################################################################################
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format

###############################################################################################################
# Functions
###############################################################################################################

# Does nothing yet, may not need to use for a while
def preprocess_features(dataframe):
  """Prepares input features from data set.

  Args:
    dataframe: A Pandas DataFrame expected to contain data
      from the data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  return dataframe

# Does nothing yet, may not need to use for a while
def preprocess_targets(dataframe):
  """Prepares target features (i.e., labels) from data set.

  Args:
    dataframe: A Pandas DataFrame expected to contain data
      from the data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  return dataframe

###############################################################################################################
# Main
###############################################################################################################

dataframe = pd.read_csv("Dataset_Github_csv.csv")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
print ("DataFrame read.")
print ("DataFrame looks like:")
print (dataframe)

# Choose the first 75%, 243 (out of 324) examples for training.
training_examples = preprocess_features(dataframe.head(243))
training_targets = preprocess_targets(dataframe.head(243))

# Choose the last 81 (out of 324) examples for validation.
validation_examples = preprocess_features(dataframe.tail(81))
validation_targets = preprocess_targets(dataframe.tail(81))

# Output dataframe subsets.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


"""
  For isolating classification information at the end of the DataFrame, you can check eg.
  if "High_grade" in dataframe.tail(1): #if the last column contains text "High-grade", same for others below. 
  if "Low-grade" in dataframe.tail(1):
  if "Normal" in dataframe.tail(1):
"""