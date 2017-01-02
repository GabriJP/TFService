import cv2
import tensorflow as tf
from os.path import join
from CNNWrappers import player_conv_net as conv_net
import numpy as np

videos = {
    'tunel': 'Other/Classes/Carreteras/tunel/tunel.mp4',
    'carretera': 'Other/Classes/Carreteras/carretera/carretera.mp4',
}


def play_cnn(meta_dataset, output):
    n_input = meta_dataset['frame_pixels']
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
    pred = conv_net(x, weights, biases, meta_dataset['shape'])

    # ----------------------------- Video capture

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videos['carretera'])

    # Launch the graph
    with tf.Session() as sess:
        saver.restore(sess, join(output, 'model'))

        while True:
            ret, img = cap.read()
            resized = cv2.resize(img, meta_dataset['shape'], interpolation=cv2.INTER_AREA)
            gray = np.asarray(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY))

            cv2.imshow('Capture', img)
            frame = gray.reshape(-1, meta_dataset['shape'][0] * meta_dataset['shape'][1])
            res = sess.run(pred, feed_dict={x: frame})
            print(meta_dataset['labels'].get(tuple(res[0])))

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    cv2.destroyAllWindows()
