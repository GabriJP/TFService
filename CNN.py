import cv2
import tensorflow as tf
import numpy as np

from os.path import join
from math import ceil


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def conv_net(x, weights, biases, frame_dimensions, layer_dropout=None):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 172, 380, 1])
    x = tf.reshape(x, shape=[-1, frame_dimensions[0], frame_dimensions[1], 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    if layer_dropout is not None:
        fc1 = tf.nn.dropout(fc1, layer_dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return tf.nn.softmax(out) if layer_dropout is None else out


def create_cnn(data_set, save_path, training_iters, learning_rate=0.001, batch_size=32, display_step=10, dropout=0.75):
    # Input layer
    x = tf.placeholder(tf.float32, [None, data_set.frame_pixels()])
    # Output layer
    y = tf.placeholder(tf.float32, [None, data_set.get_number_of_classes()])
    # Dropout Tensor
    keep_prob = tf.placeholder(tf.float32)

    # Store layers weight & bias
    weights, biases = get_weights_and_biases(data_set.get_number_of_classes(), data_set.get_frame_dimensions())

    # Construct model
    pred = conv_net(x, weights, biases, data_set.get_frame_dimensions(), keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_labels, batch_frames = data_set.next_training_batch(batch_size)
            # batch_frames = list(map((lambda frame: frame.flatten()), batch_frames))
            batch_frames = [array.flatten() for array in batch_frames]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_frames, y: batch_labels, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_frames, y: batch_labels, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))

            step += 1
        print("Optimization Finished!")

        for i in range(0, 244, batch_size):
            batch_labels, batch_frames = data_set.next_test_batch(batch_size)
            batch_frames = list(map((lambda frame: frame.flatten()), batch_frames))
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_frames, y: batch_labels, keep_prob: 1.}))

        save_path = tf.train.Saver().save(sess, join(save_path, "model"))
        print("Model saved in file: %s" % save_path)
        sess.close()

    tf.reset_default_graph()


def play_cnn(meta_dataset, output, video):
    n_input = meta_dataset['frame_pixels']
    n_classes = meta_dataset['n_classes']

    x = tf.placeholder(tf.float32, [None, n_input])
    # Store layers weight & bias
    weights, biases = get_weights_and_biases(n_classes, meta_dataset['shape'])

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Construct model
    pred = conv_net(x, weights, biases, meta_dataset['shape'])

    # ----------------------------- Video capture

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video)
    crop = meta_dataset['crop']
    shape = tuple(map(sum, zip(reversed(meta_dataset['shape']), crop)))
    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, join(output, 'model'))

        while True:
            ret, img = cap.read()
            if img is None:
                break

            resized = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)[crop[1]:, crop[0]:]
            gray = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))

            cv2.imshow('Capture', resized)
            frame = gray.reshape(-1, (crop[3]-crop[1]) * (crop[2]-crop[0]))
            res = sess.run(pred, feed_dict={x: frame})
            print(meta_dataset['labels'].get(tuple(res[0])))

            ch = 0xFF & cv2.waitKey(10)
            if ch == 27:
                break

    cv2.destroyAllWindows()
    tf.reset_default_graph()


def get_weights_and_biases(n_classes, shape):
    return {
               # 5x5 conv, 1 input, 32 outputs
               'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
               # 5x5 conv, 32 inputs, 64 outputs
               'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
               'wd1': tf.Variable(tf.random_normal([int(ceil(shape[0] / 4)) * int(ceil(shape[1] / 4)) * 128, 256])),
               # 1024 inputs, 10 outputs (class prediction)
               'out': tf.Variable(tf.random_normal([256, n_classes]))
           }, {
               'bc1': tf.Variable(tf.random_normal([64])),
               'bc2': tf.Variable(tf.random_normal([128])),
               'bd1': tf.Variable(tf.random_normal([256])),
               'out': tf.Variable(tf.random_normal([n_classes]))
           }


def multiply(elements):
    result = 1
    for element in elements:
        result *= element
    return result
