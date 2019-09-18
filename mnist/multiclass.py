##########################################################
#  Aquifi Confidential
# Copyright (c) 2017 Aquifi, Inc., All Rights Reserved
#
# THE TERMS OF USE ARE SUBJECT TO THE PREVAILING LICENSING
# AGREEMENT. THIS FILE MAY NOT BE COPIED
# NOR DISTRIBUTED TO ANY OTHER PARTY.
##########################################################

import tensorflow as tf
import numpy as np

###############
# Mnist Network setup
###############

# Parameters
dropout = tf.placeholder_with_default(tf.constant(1.), shape=[], name='dropout')

# Input
img_d = 1 # Using gray scale images
image = tf.placeholder(tf.uint8, [None, None, None, img_d], name='image')
num_labels = 10
label = tf.placeholder(tf.float32, [None, num_labels], name='label')

# Preprocessing
with tf.variable_scope('preprocessing'):
    #image_prep = tf.reshape(tf.image.resize_images(image, [256, 256]), [-1, 256, 256, img_d])
    image_prep = tf.reshape(tf.image.resize_images(image, [28, 28]), [-1, 28, 28, img_d])
    #image_prep = image

# Convolutional layer 1
with tf.variable_scope('conv1'):
    weights = tf.get_variable(name='weights', shape=[3, 3, img_d, 2], initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(image_prep, weights, [1, 1, 1, 1], padding='VALID') # Stride is always [1, stride, stride, 1]
    biases = tf.get_variable(name='biases', shape=[2], initializer=tf.constant_initializer(0.0))
    out1 = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(out1, name='conv1')
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    #pool1 = conv1

# Fully-connected layer 1
with tf.variable_scope('fc1'):
    all_nodes = 12 * 12 * 2
    pool1_flat = tf.reshape(pool1, shape=[-1, all_nodes])
    weights = tf.get_variable(name='weights', shape=[all_nodes, num_labels], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases', shape=[num_labels], initializer=tf.constant_initializer(0.0))
    fc = tf.nn.xw_plus_b(pool1_flat, weights, biases)
    fc1 = tf.nn.dropout(tf.check_numerics(fc, 'Logits were NON-numeric (NaN or Inf)'), dropout, name='fc1')

# Prediction
prediction = tf.nn.softmax(fc1, name='output')

# Optimization
with tf.name_scope('optimization'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc1, labels=label), name='loss')
    learningRate = tf.placeholder(tf.float32, name='learningRate')
    optimizer = tf.train.AdagradOptimizer(learningRate).minimize(loss, name="optimizer")

#  Initializer
init = tf.global_variables_initializer()

#  Saver
saver = tf.train.Saver(tf.trainable_variables())


###############
# Test training
###############

if __name__ == "__main__":
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        # Initialize model
        print('Initializing model')
        sess.run(init)

        # Run one iteration
        img = np.ones([1, 28, 28, img_d], np.uint8) * 255
        l = np.asarray([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32)
        print('start training: ')
        sess.run([optimizer], feed_dict={image: img, label: l, learningRate: 0.02})
        print('pred: ', sess.run(prediction, feed_dict={image: img, label: l})[0])
        print('loss: %f' % sess.run(loss, feed_dict={image: img, label: l}))

        # Save graph
        saver.save(sess, 'multiclass')



