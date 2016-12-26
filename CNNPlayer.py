import cv2
import tensorflow as tf
from os.path import join
from DataSet import DataSet
from CNNWrappers import player_conv_net as conv_net


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
    pred = conv_net(x, weights, biases)

    # ----------------------------- Video capture

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(videos['tunel'])

    labels = {tuple(y): x for x, y in meta_dataset['labels'].items()}

    # Launch the graph
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(join(output, 'model.meta'))
        new_saver.restore(sess, tf.train.latest_checkpoint(output))

        while True:
            ret, img = cap.read()
            # cropped = resized[0:180, 70:250]
            # resized64 = cv2.resize(cropped, (128, 128), interpolation = cv2.INTER_AREA)
            # gray = np.asarray(cv2.cvtColor(resized64, cv.CV_RGB2GRAY))

            frame = DataSet.process_frame(img, meta_dataset['shape'], (0, 0, 140, 80)).reshape([1, 11200])
            cv2.imshow('Capture', img)
            res = sess.run(pred, feed_dict={x: frame})
            print(labels[tuple(res[0])])

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

    cv2.destroyAllWindows()
