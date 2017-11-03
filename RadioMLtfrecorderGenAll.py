# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# In[1]:
import numpy as np
import tensorflow as tf
import pickle
# In[2]:
# Load the dataset You will need to separately download or generate this file

with open("./RML2016.10a_dict.dat", 'rb') as temp:
    Xd = pickle.load(temp,encoding='latin1')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j],Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)
del(Xd)
# In[2]:
# Partition the data(Normalized IQ and Ap) into training and test sets of the form we can train/test on
#   while keeping SNR and Mod labels handy for each example

ComplexX = X[:,0,:]+1j*X[:,1,:]
Amp =   np.absolute(ComplexX)
Amp = Amp/(np.reshape(np.sqrt(np.mean(Amp**2,axis=1)),(-1,1))*np.ones([1,128]))
Phase = np.angle(ComplexX)/np.pi
X_IQ = np.zeros([X.shape[0],X.shape[2],X.shape[1]])
X_IQ[:,:,0] = np.real(Amp*np.exp(1j*Phase*np.pi))
X_IQ[:,:,1] = np.imag(Amp*np.exp(1j*Phase*np.pi))
X_AP = np.zeros([X.shape[0],X.shape[2],X.shape[1]])
X_AP[:,:,0] = Amp
X_AP[:,:,1] = Phase
del(ComplexX,Amp,Phase)

np.random.seed(2017)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5)
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))

train_SNRs = list(map(lambda x: lbl[x][1], train_idx))
test_SNRs = list(map(lambda x: lbl[x][1], test_idx))

X_train_IQ = X_IQ[train_idx]
X_train_AP = X_AP[train_idx]
X_test_IQ = X_IQ[test_idx]
X_test_AP = X_IQ[test_idx]

Y_train = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

#
in_shp1 = list(X_train_IQ.shape[1:])
in_shp2 = list(X_train_AP.shape[1:])

print(X_train_IQ.shape,X_train_AP.shape,in_shp1,in_shp2)
classes = mods

# In[3]:save training data and testing data to TFrecords
TFwriter = tf.python_io.TFRecordWriter("RMLtrainAll.tfrecords")
numexample = 0
for i in range(Y_train.shape[0]):
    label = int(Y_train[i,])
    snr = int(train_SNRs[i])
    sig_iq = X_train_IQ[i,:].tobytes()
    sig_ap = X_train_AP[i,:].tobytes()
    example = tf.train.Example(features=tf.train.Features(
    feature={"label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
    "snr":tf.train.Feature(int64_list = tf.train.Int64List(value=[snr])),
    "sig_iq":tf.train.Feature(bytes_list = tf.train.BytesList(value=[sig_iq])),
    "sig_ap":tf.train.Feature(bytes_list = tf.train.BytesList(value=[sig_ap]))
    }) )
    TFwriter.write(example.SerializeToString())
    numexample = numexample+1
print('number of training example:%d'%numexample)
TFwriter.close()

TFwriter = tf.python_io.TFRecordWriter("RMLtestAll.tfrecords")

numexample = 0
for i in range(Y_test.shape[0]):
    label = int(Y_test[i,])
    snr = int(test_SNRs[i])
    sig_iq = X_test_IQ[i,:].tobytes()
    sig_ap = X_test_AP[i,:].tobytes()
    example = tf.train.Example(features=tf.train.Features(
    feature={"label":tf.train.Feature(int64_list = tf.train.Int64List(value=[label])),
    "snr":tf.train.Feature(int64_list = tf.train.Int64List(value=[snr])),
    "sig_iq":tf.train.Feature(bytes_list = tf.train.BytesList(value=[sig_iq])),
    "sig_ap":tf.train.Feature(bytes_list = tf.train.BytesList(value=[sig_ap]))
    }) )
    TFwriter.write(example.SerializeToString())
    numexample = numexample+1
print('number of testing example:%d'%numexample)
TFwriter.close()
