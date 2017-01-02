import tensorflow as tf


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def player_conv_net(x, weights, biases, frame_dimensions):
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

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return tf.nn.softmax(out)


def creator_conv_net(input_layer, layer_weights, layer_biases, layer_dropout, frame_dimensions):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 172, 380, 1])
    input_layer = tf.reshape(input_layer, shape=[-1, frame_dimensions[0], frame_dimensions[1], 1])

    # Convolution Layer
    conv1 = conv2d(input_layer, layer_weights['wc1'], layer_biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, layer_weights['wc2'], layer_biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, layer_weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, layer_weights['wd1']), layer_biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, layer_dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, layer_weights['out']), layer_biases['out'])
    return out
