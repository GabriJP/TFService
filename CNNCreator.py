import tensorflow as tf
from os.path import join


def create_cnn(data_set, learning_rate=0.001, training_iters=3000, batch_size=32, display_step=10, dropout=0.75,
               save_path='.'):
    # Input layer
    x = tf.placeholder(tf.float32, [None, data_set.frame_pixels()])
    # Output layer
    y = tf.placeholder(tf.float32, [None, data_set.get_number_of_classes()])
    # Dropout Tensor
    keep_prob = tf.placeholder(tf.float32)

    # Create some wrappers for simplicity
    def conv2d(layer, layer_weights, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        layer = tf.nn.conv2d(layer, layer_weights, strides=[1, strides, strides, 1], padding='SAME')
        layer = tf.nn.bias_add(layer, b)
        return tf.nn.relu(layer)

    def maxpool2d(tensor, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    # Create model
    def conv_net(layer, layer_weights, layer_biases, layer_dropout):
        # Reshape input picture
        # x = tf.reshape(x, shape=[-1, 172, 380, 1])
        layer = tf.reshape(layer, shape=[-1, 140, 80, 1])

        # Convolution Layer
        conv1 = conv2d(layer, layer_weights['wc1'], layer_biases['bc1'])
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

    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([35 * 20 * 128, 256])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([256, data_set.get_number_of_classes()]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([64])),
        'bc2': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([256])),
        'out': tf.Variable(tf.random_normal([data_set.get_number_of_classes()]))
    }

    # Construct model
    pred = conv_net(x, weights, biases, keep_prob)

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
            batch_frames = list(map((lambda frame: frame.flatten()), batch_frames))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_frames, y: batch_labels, keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_frames, y: batch_labels, keep_prob: 1.})
                print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc))

                if acc == 1:
                    break

            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 245 OCT test images
        for i in range(0, 244, batch_size):
            batch_labels, batch_frames = data_set.next_test_batch(batch_size)
            batch_frames = list(map((lambda frame: frame.flatten()), batch_frames))
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_frames, y: batch_labels, keep_prob: 1.}))

        save_path = tf.train.Saver().save(sess, join(save_path, "model.ckpt"))
        print("Model saved in file: %s" % save_path)
