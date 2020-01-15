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

def preprocess_features(dataframe):
    processed_features = dataframe.copy()
    processed_features = processed_features.drop(labels='High-grade tumor-1', axis=1) # Remove targets
    return processed_features

def preprocess_targets(dataframe):
    processed_targets = dataframe.pop('High-grade tumor-1')  # Remove targets
    return processed_targets

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def input_fn_train (dataframe, name_target):
    return tf.estimator.inputs.pandas_input_fn(
      x=dataframe,
      y=dataframe[name_target],
      batch_size = 32,
      num_epochs = 5,
      shuffle = True,
      queue_capacity = 2000,
      num_threads = 1
    )

#mainline
dataframe = pd.read_csv("data.csv")
dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
print ("DataFrame read.")
#Set training
training_features = preprocess_features(dataframe.head(243))
training_targets = preprocess_targets(dataframe.head(243))
training_dataframe = dataframe.head(243)
#Set validation
print ("Training set.")
validation_features = preprocess_features(dataframe.tail(81))
validation_targets = preprocess_targets(dataframe.tail(81))
print ("Validation set.")
#Build model
model = tf.compat.v2.estimator.DNNClassifier(
    feature_columns=construct_feature_columns(training_features),
    # Using rule of thumb for hidden layers, square root of the features. We should play around with the testing of this.
    hidden_units=[37, 6], # The format is hidden_units = [numberOfNodesInFirstLayer, numberOfNodesInSecondLayer, etc.]
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))
batch_size = 5
model.train(input_fn=lambda : input_fn_train(training_features, 'High-grade tumor-1'))

