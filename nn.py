import tensorflow as tf
import pickle
import numpy as np
import time


X_train = pickle.load(open("X_train.pickle", "rb"))
print("X", len(X_train))
y_train = pickle.load(open("y_train.pickle", "rb"))
print("y", len(y_train), "Shape", tf.shape(y_train))
X_test = pickle.load(open("X_test.pickle", "rb"))
print("X", len(X_test))
y_test = pickle.load(open("y_test.pickle", "rb"))
print("y", len(y_test))
print("DATA LOADED")

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

dropout_prob = 0.5


def conv_net(x, weights, biases, dropout):
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])
    conv2 = maxpool2d(conv2)

    conv3 = conv2d(conv2, weights['layer_3'], biases['layer_3'])
    conv4 = conv2d(conv3, weights['layer_4'], biases['layer_4'])
    conv4 = maxpool2d(conv4)

    fc1 = tf.reshape(conv4, [-1, weights['dense_1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['dense_1']), biases['dense_1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout_prob)

    fc2 = tf.add(tf.matmul(fc1, weights['dense_2']), biases['dense_2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout_prob)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return out

training_epochs = 10

init = tf.global_variables_initializer()
'''
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        total_batch = int(n_train / batch_size)

        # shuffle data index for each epoch
        rand_idx = np.random.permutation(n_train)

        for i in range(total_batch):
            offset = i * batch_size
            off_end = offset + batch_size
            batch_idx = rand_idx[offset:off_end]

            batch_x = X_train[batch_idx]
            batch_y = y_train_ohe[batch_idx]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout_prob})

        cost_ts, acc_ts = sess.run([cost, accuracy], feed_dict={x: X_test, y: y_test_ohe, keep_prob: 1.})

        print("Cost:  {:.5f}  | Accuracy:  {:.5f}".format(cost_ts, acc_ts))

    save_path = saver.save(sess, "models/model.ckpt")
    print("Training Complete! Model saved in file: %s" % save_path)
'''