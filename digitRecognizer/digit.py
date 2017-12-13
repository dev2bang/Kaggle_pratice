import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


X_train = (train.ix[:,1:].values).astype('float32') # all pixel values
y_train = train.ix[:,0].values.astype('int32')       # only labels i.e targets digits
X_test = test.values.astype('float32')
# Convert train data set to (num_images, img_rows, img_cols) format
# X_train = X_train.reshape(X_train.shape[0], 28, 28)

# expand 1 more dimention as 1 for colour channel gray
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#print(X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
#print(X_test.shape)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px

y_train = to_categorical(y_train)
num_classes = y_train.shape[1]

# fix random seed for reproducibility
seed = 43
np.random.seed(seed)

# Create Model
model = Sequential()
model.add(Lambda(standardize, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print("input shape ", model.input_shape)
print("output shape ", model.output_shape)

model.compile(optimizer=RMSprop(lr=0.001),
        loss='categorical_crossentyopy',
        metrics=['accuracy'])


gen = image.ImageDataGenerator()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches=gen.flow(X_val, y_val, batch_size=64)

history=model.fit_generator(batches, batches.n, nb_epoch=1,
        validation_data=val_batches, nb_val_samples=val_batches.n)

history_dict = history.hisotry
history_dict.keys()

loss_values = history_dict['loss']
loss_values = history_dict['loss']

