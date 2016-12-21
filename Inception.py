# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception V4 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.nets import inception
import tensorflow as tf
import numpy as np


def one_hot(label_list, n):
    label_array = np.array(label_list).flatten()
    o_h = np.zeros((len(label_array), n))
    o_h[np.arange(len(label_array)), label_array - 1] = 1
    return o_h


def nn(data_set):
    n_classes = data_set.get_number_of_classes()
    classes = data_set.get_classes()
    labels = one_hot(list(range(n_classes)), n_classes)

    # Parameters
    learning_rate = 0.001
    training_iters = 3000
    batch_size = 32
    display_step = 10

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    # Network Parameters
    dropout = 0.75  # Dropout, probability to keep units

    # TODO Backwards compatibility
    # noinspection PyCompatibility
    x = tf.placeholder(tf.float32, [batch_size, *data_set.get_frame_dimensions()], 3)
    pred, endpoints = inception.inception_v3(x, data_set.get_number_of_classes())
    y = tf.placeholder(tf.float32, [None, data_set.get_number_of_classes()])

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_y = data_set.next_training_batch(batch_size)
            batch_x = [labels[classes.index(l)] for l in batch_x]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_y, y: batch_x, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_y, y: batch_x, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 245 OCT test images

        for i in range(0, 244, 30):
            batch_x, batch_y = data_set.next_test_batch(batch_size)
            batch_x = [labels[classes.index(l)] for l in batch_x]
            batch_y = [array.flatten() for array in batch_y]
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_y, y: batch_x, keep_prob: 1.}))

        # Save the variables to disk.
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in file: %s" % save_path)
