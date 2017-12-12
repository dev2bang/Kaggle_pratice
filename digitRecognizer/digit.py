import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32')       # only labels i.e targets digits
X_test = test.values.astype('float32')
# Convert train data set to (num_images, img_rows, img_cols) format
# X_train = X_train.reshape(X_train.shape[0], 28, 28)

# expand 1 more dimention as 1 for color channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
print(X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print(X_test.shape)

