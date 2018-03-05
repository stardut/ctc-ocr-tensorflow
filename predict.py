# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import cv2
from model import Model
from data import Data
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

input_size = 28
num_class = 10 + 1
num_layers = 2
seq_len = 28
batch_size = 1
num_units = 256
word_size = 8

model_dir = 'model/'

if not os.path.exists(model_dir):
    print('no model.')
    exit()

if sys.argv[1] == None:
    print('need a image path.')
    exit()

def normalize(array):
    return (array - array.min()) / (array.max() - array.min())

img = cv2.imread(sys.argv[1], 0)
img = cv2.resize(img, (seq_len*word_size, input_size))
img = normalize(img)
img = np.transpose(img)
seq_lens = np.ones(batch_size) * (seq_len * word_size)
inputs = []
inputs.append(img)

model = Model(input_size=input_size,
              num_class=num_class,
              batch_size=batch_size,
              num_units=num_units,
              num_layers=num_layers)
data = Data()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, 'model/model')
    feed = {
        model.inputs : inputs,
        model.seq_len : seq_lens,
        model.keep_prob : 1.0
    }

    decode = sess.run(model.decoded, feed_dict=feed)
    pre = data.decode_sparse_tensor(decode[0])
    print('predict: ' + pre[0])
    cv2.imshow('res', np.transpose(img))
    cv2.waitKey(0)







