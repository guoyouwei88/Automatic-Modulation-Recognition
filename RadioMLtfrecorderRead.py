# -*- coding: utf-8 -*-
import tensorflow as tf

# In[1]: read training data and testing data from TFrecords
def read_data(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'label': tf.FixedLenFeature([], tf.int64),
          'snr': tf.FixedLenFeature([], tf.int64),
          'sig_iq':tf.FixedLenFeature([], tf.string),
          'sig_ap':tf.FixedLenFeature([], tf.string),
          })

    sig_iq = tf.decode_raw(features['sig_iq'], tf.float64)
    sig_iq = tf.reshape(sig_iq, [-1, 128, 2]) 
    sig_iq = tf.cast(sig_iq, tf.float32)
    sig_iq = tf.image.resize_images(sig_iq, [128, 2])

    sig_ap = tf.decode_raw(features['sig_ap'], tf.float64)
    sig_ap = tf.reshape(sig_ap, [-1, 128, 2])
    sig_ap = tf.cast(sig_ap, tf.float32)
    sig_ap = tf.image.resize_images(sig_ap, [128, 2])
    
    label = tf.cast(features['label'], tf.int32)
    snr = tf.cast(features['snr'], tf.int32)
    return sig_iq,sig_ap,label,snr

def read_data_batch(file_queue, batch_size):
    sig_iq,sig_ap,label,snr = read_data(file_queue)
    capacity = 3 * batch_size
    sig_iq_batch,sig_ap_batch,label_batch= tf.train.batch([sig_iq, sig_ap, label], batch_size=batch_size, capacity=capacity, num_threads=1000)
    return sig_iq_batch,sig_ap_batch,label_batch

train_data_filename_queue = tf.train.string_input_producer(["D:\PythonScript\AMR\RMLtrainAll.tfrecords"])

train_sigiqs, train_sigaps,train_labels = read_data_batch(train_data_filename_queue, batch_size=100)

test_data_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("D:\PythonScript\AMR\RMLtestAll.tfrecords"))
test_sigiqs, test_sigaps,test_labels = read_data_batch(test_data_filename_queue, 100)
    
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        exampleIQ, exampleAP, l = sess.run([train_sigiqs,train_sigaps,train_labels])#在会话中取出iq/ap/label
    coord.request_stop()
    coord.join(threads)