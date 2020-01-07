import tensorflow as tf
import pickle
import numpy as np
import time

train_x = pickle.load(open("X_train.pickle", "rb"))
print("X", len(train_x))
train_y = pickle.load(open("y_train.pickle", "rb"))
print("y", len(train_y), "Shape", tf.shape(train_y))
test_x = pickle.load(open("X_test.pickle", "rb"))
print("X", len(test_x))
test_y = pickle.load(open("y_test.pickle", "rb"))
print("y", len(test_y))
print("DATA LOADED")

n_classes = 3
batch_size = 32
print("Building placeholders")
x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.9
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(x):
    print("Building Network")
    weights = {'W_conv1': tf.Variable(tf.random_normal([11, 11, 3, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),  # 1024 is the number of nodes
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    # x = tf.reshape(x, shape=[-1, 32, 23, 1])

    print("Building conv1")
    print("weight 1", weights['W_conv1'])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']))
    conv1 = maxpool2d(conv1)
    print("Building conv2")
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']))
    conv2 = maxpool2d(conv2)
    print("Building fully connected layer")
    fc = tf.reshape(conv2, [-1, 64 * 64 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']))
    fc = tf.nn.dropout(fc, keep_rate)
    print("Building output layer")
    output = tf.matmul(fc, weights['out'])

    return output

import numpy as np

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def train_neural_network(x):
    print("Starting training")
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print("building training section")
    hm_epochs = 3
    with tf.Session() as sess:
        print("Starting session")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(train_x) / batch_size)):
                epoch_x, epoch_y = next_batch(batch_size, train_x, train_y) # mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            ep = epoch + 1
            print('Epoch', ep, '/', hm_epochs, '-------------- Loss:', epoch_loss, '        Accuracy:',
                  accuracy.eval({x: test_x, y: test_y}))
    '''
        for epoch in range(hm_epochs):
            print(epoch + 1, "-------")
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, '--------------------Loss:', epoch_loss, 'Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    '''


Start_time = time.time()
train_neural_network(x)
print("EXECUTION TIME: ", int((time.time() - Start_time)//60), ":", int((time.time() - Start_time)%60))