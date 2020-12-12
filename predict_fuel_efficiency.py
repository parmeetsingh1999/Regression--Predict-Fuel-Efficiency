# Importing libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)
np.set_printoptions(precision = 3, suppress = True)

# Loading the dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
raw_dataset = pd.read_csv(url, names = column_names, na_values = '?', comment = '\t', sep = ' ', skipinitialspace = True)
dataset = raw_dataset.copy()
print(dataset.head())

# Clean the data

print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.dtypes)
dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(dataset, prefix = '', prefix_sep = '')
print(dataset.head())

# Splitting the data into train and test

train_data = dataset.sample(frac = 0.8, random_state = 0)
test_data = dataset.drop(train_data.index)
print(train_data.shape)
print(test_data.shape)

# Inspect the data

sns.pairplot(train_data[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind = "kde")
print(train_data.describe())

# Split features from labels

train_features = train_data.copy()
test_features = test_data.copy()
train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")

# Normalization

print(train_data.describe().transpose()[["mean", "std"]])

# Normalization layer

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
first = np.array(train_features[:1])
with np.printoptions(precision = 2, suppress = True):
  print("First example - ", first)
  print("Normalized - ", normalizer(first).numpy())

# Linear Regression

horsepower = np.array(train_features["Horsepower"])
horsepower_normalizer = preprocessing.Normalization(input_shape = [1, ])
horsepower_normalizer.adapt(horsepower)
horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units = 1)])
print(horsepower_model.summary())
print(horsepower_model.predict(horsepower[:10]))
horsepower_model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1), loss = "mean_absolute_error")
history = horsepower_model.fit(train_features["Horsepower"], train_labels, epochs = 100, verbose = 0, validation_split = 0.2)
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
print(hist.tail())

def plotLoss(history):
  plt.plot(history.history["loss"], label = "loss")
  plt.plot(history.history["val_loss"], label = "val_loss")
  plt.ylim([0, 10])
  plt.xlabel("Epoch")
  plt.ylabel("Error [MPG]")
  plt.legend()
  plt.grid(True)
  
plotLoss(history)
test_results = {}
test_results["horsepower_model"] = horsepower_model.evaluate(test_features["Horsepower"], test_labels, verbose = 0)
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plotHorsepower(x, y):
  plt.scatter(train_features["Horsepower"], train_labels, label = "Data")
  plt.plot(x, y, color = 'k', label = "Predictions")
  plt.xlabel("Horsepower")
  plt.ylabel("MPG")
  plt.legend()
  
plotHorsepower(x, y)

# Multiple inputs

linear_model = tf.keras.Sequential([normalizer, layers.Dense(units = 1)])
print(linear_model.predict(train_features[:10]))
print(linear_model.layers[1].kernel)
linear_model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1), loss = "mean_absolute_error")
history = linear_model.fit(train_features, train_labels, epochs = 100, verbose = 0, validation_split = 0.2)
plotLoss(history)
test_results["linear_model"] = linear_model.evaluate(test_features, test_labels, verbose = 0)

# DNN Regression

def buildAndCompileModel(norm):
  model = keras.Sequential([
                            norm, 
                            layers.Dense(64, activation = "relu"),
                            layers.Dense(64, activation = "relu"),
                            layers.Dense(1)
                            ])
  
  model.compile(loss = "mean_absolute_error", optimizer = tf.keras.optimizers.Adam(0.001))
  return model

# One Variable

dnn_horsepower_model = buildAndCompileModel(horsepower_normalizer)
print(dnn_horsepower_model.summary())
history = dnn_horsepower_model.fit(train_features["Horsepower"], train_labels, validation_split = 0.2, verbose = 0, epochs = 100)
plotLoss(history)
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plotHorsepower(x, y)
test_results["dnn_horsepower_model"] = dnn_horsepower_model.evaluate(test_features["Horsepower"], test_labels, verbose = 0)

# Full Model

dnn_model = buildAndCompileModel(normalizer)
print(dnn_model.summary())
history = dnn_model.fit(train_features, train_labels, validation_split = 0.2, verbose = 0, epochs = 100)
plotLoss(history)
test_results["dnn_model"] = dnn_model.evaluate(test_features, test_labels, verbose = 0)

# Performance

pd.DataFrame(test_results, index = ["Mean absolute error [MPG]"]).T

# Make Predictions

test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect = "equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction error [MPG]")
_ = plt.ylabel("Count")

dnn_model.save("dnn_model")
reloaded = tf.keras.models.load_model("dnn_model")
test_results["reloaded"] = reloaded.evaluate(test_features, test_labels, verbose = 0)
pd.DataFrame(test_results, index = ["Mean absolute error [MPG]"]).T
