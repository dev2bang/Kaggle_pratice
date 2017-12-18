from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Activation, Dropout, Dense

import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (train.ix[:,:].values).astype('float32')

y_train = np_utils.to_categorical(labels)

scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
np_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(128,input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(np_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)

preds = model.predict_classes(X_test, verbose=0)

pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"label":preds}).to_csv("DR2.csv", index=False, header=True)
