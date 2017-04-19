# Import all the things we need
# by setting env variables before Keras import you can set up which backend and which GPU it use
# %matplotlib inline
import os, random
os.environ["KERAS_BACKEND"] = "theano"
# os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
# import theano as th
# import tensorflow as tf
# import theano.tensor as th
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras

# Load the dataset
# You will need to separately download or generate this file
Xd = cPickle.load(open("RML2016.10a_dict.dat", 'rb'))
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j],Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)

# Partition the data
#  into training and test sets of the form we can train/test on
#   while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test = X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

#
in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods

# Build VT-CNN2 Neural Net model using Keras primitives --
#   - Reshape [N,2,128] to [N,1,2,128] on input
#   - Pass through 2 2DConv/ReLu layers
#   - Pass through 2 Dense layers (ReLu and Softmax)
#   - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate(%)
model = models.Sequential()
model.add(Reshape([1]+in_shp,input_shape=in_shp))
model.add(ZeroPadding2D(padding=(0, 2), dim_ordering='th'))
model.add(Convolution2D(256, 1, 3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform', dim_ordering='th'))
model.add(Dropout(dr))
model.add(ZeroPadding2D(padding=(0, 2), dim_ordering='th'))
model.add(Convolution2D(80, 2, 3, border_mode='valid', activation="relu", name="conv2", init='glorot_uniform', dim_ordering='th'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), init='he_normal', name='dense2'))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# Set up some params
nb_epoch = 100  # number of epochs to train on
batch_size = 128  # training batch size

# perform training
#   - call the main training loop in keras for network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
                    Y_train,
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    show_accuracy=False,
                    verbose=2,
                    validation_data=(X_test,Y_test),
                    callbacks = [
                        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',verbose=0,save_best_only=True, mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)

# Show simple version of performance
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0, batch_size=batch_size)
print score
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
