import numpy as np
import cv2
import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 172, 380, 1])
    x = tf.reshape(x, shape=[-1, 140, 80, 1])

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


def play_cnn(meta_dataset):
    width, height = meta_dataset['shape']
    n_input = width * height
    n_classes = meta_dataset['n_classes']

    x = tf.placeholder(tf.float32, [None, n_input])
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([35 * 20 * 128, 256])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([256, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Construct model
    pred = conv_net(x, weights, biases)

    # ----------------------------- Video capture

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('Other/Classes/Carreteras/tunel/tunel.mp4')

    labels = {tuple(y): x for x, y in meta_dataset['labels'].items()}

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, "Other/Output/model.ckpt")

        while True:
            ret, img = cap.read()
            resized = cv2.resize(img, meta_dataset['shape'], interpolation=cv2.INTER_AREA)
            # cropped = resized[0:180, 70:250]
            # resized64 = cv2.resize(cropped, (128, 128), interpolation = cv2.INTER_AREA)
            # gray = np.asarray(cv2.cvtColor(resized64, cv.CV_RGB2GRAY))
            gray = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))

            cv2.imshow('Capture', img)
            cv2.waitKey(35)
            frame = gray.reshape(-1, 11200)
            res = sess.run(pred, feed_dict={x: frame})
            print(labels[tuple(res[0])])

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

            cv2.destroyAllWindows()
