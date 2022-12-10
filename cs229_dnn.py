# -*- coding: utf-8 -*-
"""cs229_dnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-o_doQK9w3AcUo65uNkZ6TlFoDajx4wi
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss

# Commented out IPython magic to ensure Python compatibility.
#this mounts your Google Drive to the Colab VM.
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# enter the foldername in the Shared Google Drive
FOLDERNAME = 'Shared drives/CS 229 Project'
assert FOLDERNAME is not None, "[!] Enter the foldername."

# now that we've mounted your Drive, this ensures that
# the Python interpreter of the Colab VM can load
# python files from within it.
import sys
sys.path.append('/content/drive/{}'.format(FOLDERNAME))

# %cd /content/drive/$FOLDERNAME/

# load data into vars
train_data = pd.read_csv('./train_points_with_point_history.csv')
eval_data = pd.read_csv('./eval_points_with_point_history.csv')
test_data = pd.read_csv('./test_points_with_point_history.csv')

x_train = train_data.iloc[:, 1:512].to_numpy()
y_train = train_data.iloc[:, 512].to_numpy()
# print(y_train[-1])
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)

x_val = eval_data.iloc[:, 1:512].to_numpy()
y_val = eval_data.iloc[:, 512].to_numpy()
# print(len(eval_data.iloc[:].to_numpy()[0][512]))
print('x_val.shape: ', x_val.shape)
print('y_val.shape: ', y_val.shape)

x_test = test_data.iloc[:, 1:512].to_numpy()
y_test = test_data.iloc[:, 512].to_numpy()
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

n_classes = 1
n_features = x_train.shape[1]

model = Sequential()
model.add(tf.keras.layers.Dense(units=40, activation='tanh'))
model.add(tf.keras.layers.Dense(units=60, activation='tanh'))
model.add(tf.keras.layers.Dense(units=80, activation='tanh'))
model.add(tf.keras.layers.Dense(units=100, activation='tanh'))
model.add(tf.keras.layers.Dense(units=120, activation='tanh'))
model.add(tf.keras.layers.Dense(units=256, activation='tanh'))
model.add(tf.keras.layers.Dense(units=256, activation='tanh'))
model.add(Dense(n_classes, activation = 'sigmoid', input_dim = n_features))
model.compile(optimizer='adam', loss='binary_crossentropy')
#model.summary()
model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))

# y_eval_pred = model.evaluate(x_eval)
# y_eval_out = y_eval >= 0.5

y_pred = model.predict(x_val)
y_out = y_pred >= 0.5

accuracy_score(y_val, y_out)

brier_score_loss(y_val, y_pred)

from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibrationDisplay

prob_true, prob_pred = calibration_curve(y_val, y_pred, n_bins=10)
disp = CalibrationDisplay(prob_true, prob_pred, y_pred)
disp.plot(name="Calibration Curve for DNN on Test Set")

from sklearn.utils.validation import column_or_1d
def calibration_error(y_true, y_prob, n_bins=10):
  y_true = column_or_1d(y_true)
  y_prob = column_or_1d(y_prob)

  bins = np.linspace(0.0, 1.0, n_bins + 1)
  binids = np.searchsorted(bins[1:-1], y_prob)

  bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
  bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
  bin_total = np.bincount(binids, minlength=len(bins))

  nonzero = bin_total != 0

  ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_test)))
  return ece

calibration_error(y_val, y_pred)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
# y_pred1 = model.predict(X_test)
# y_pred = np.argmax(y_pred1, axis=1)

# Print f1, precision, and recall scores
# print(confusion_matrix(y_val, y_out))
print(precision_score(y_val, y_out))
print(recall_score(y_val, y_out))
print(f1_score(y_val, y_out))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


cm = confusion_matrix(y_val, y_out)
labels = ['0', '1']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Optimized DNN Model')
plt.show()

"""Test Set Prediction"""

y_test_pred = model.predict(x_test)
y_test_out = y_test_pred >= 0.5

prob_test_true, prob_test_pred = calibration_curve(y_test, y_test_pred, n_bins=10)
disp = CalibrationDisplay(prob_test_true, prob_test_pred, y_test_pred)
disp.plot(name="Calibration Curve for DNN on Test Set")

accuracy_score(y_test, y_test_out)

brier_score_loss(y_test, y_test_out)

`calibration_error(y_test, y_test_pred)

print(precision_score(y_test, y_test_out))
print(recall_score(y_test, y_test_out))
print(f1_score(y_test, y_test_out))

cm_test = confusion_matrix(y_test, y_test_out)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for DNN on Test Set')
plt.show()
