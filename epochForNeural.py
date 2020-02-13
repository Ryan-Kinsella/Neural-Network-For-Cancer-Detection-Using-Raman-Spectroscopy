import numpy as np
import pandas as pd
import tensorflow as tf
import time
from sklearn.metrics import accuracy_score as acc
print(tf.version) # tensorflow 2.0
pd.options.display.max_rows = 8 # will use 8 by default for count, mean, std ... max
pd.options.display.max_columns = 9
pd.options.display.float_format = '{:.6f}'.format
pd.set_option('mode.chained_assignment', None)
df = pd.read_csv("Dataset_Github_Labeled.csv")

# change y in the csv file to be assigned to one of three classes: High-grade, Low-grade, Normal
for i in range (0,324): # 0 - 323, same size as x
    if df['class'][i].startswith('High-grade'):  # if the last column contains text "High-grade", etc below.
        df['class'][i] = 'High-grade'
    elif df['class'][i].startswith('Low-grade'):
        df['class'][i] = 'Low-grade'
    elif df['class'][i].startswith('Normal'):
        df['class'][i] = 'Normal'


# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
# x represents attributes, y represents class label
training, validation, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))]) # 60% test, 20% validation, 20% test split.
x_train = training.drop(['class'], axis=1)
y_train = training['class']
x_validation=validation.drop(['class'], axis=1)
y_validation=validation['class']
x_test=test.drop(['class'], axis=1)
y_test=test['class']
# Encode class label y to be 0, 1, or 2
from sklearn.preprocessing import LabelEncoder
lbl_encoder = LabelEncoder()
y_train= lbl_encoder.fit_transform(y_train)
y_test= lbl_encoder.fit_transform(y_test)
y_validation= lbl_encoder.fit_transform(y_validation)
del training, validation, test # clear memory of variables not needed


def construct_feature_columns(input_features_DataFrame):
  """Construct the TensorFlow Feature Columns.
  Args:
    input_features: DataFrame with the names of the numerical input features to use.
  Returns:
    A set of tf numeric feature columns
  """
  tensorSet = ([])
  for elem in input_features_DataFrame:
    tensorSet.append(tf.feature_column.numeric_column(key=str(elem), dtype=tf.dtypes.float64) ) # where elem is a str feature label
  return tensorSet

# Create the input function for training + evaluation. boolean = True for training.
def input_fn(features, labels, training=True, batch_size=32 ):
    dataf = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataf = dataf.shuffle(100).repeat() # shuffle ~<half the data
    return dataf.batch(batch_size=batch_size)

 #Construct feature columns, which is a list of tensors with the feature names, shown below.
feature_columns=construct_feature_columns(df.drop(['class'], axis=1).head(0))

len(feature_columns)

# tf.keras.backend.set_floatx('float64')
timer_start = time.time()
learning_rate=0.001
if (tf.__version__[0] == '2'):
    optimizer_adam= tf.optimizers.Adam(learning_rate=learning_rate)
else:
    optimizer_adam= tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
hidden_units=[37,30,19]
print("Building model: Hidden units = " , hidden_units)
model=tf.estimator.DNNClassifier(hidden_units=hidden_units, feature_columns=feature_columns, n_classes=3,
        optimizer=optimizer_adam)
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

validation_predictions = list(model.predict(input_fn=lambda: input_fn(features=x_validation, labels=y_validation, training=False), yield_single_examples=True))


validation_predictions_class_ids = [item['class_ids'][0] for item in validation_predictions]

print(validation_predictions_class_ids)
