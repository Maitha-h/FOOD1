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

'''

def weight_matrix(shape):
    return tf.Variable(tf.compat.v1.truncated_normal(shape, stddev=0.1))


def biases_matrix(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))

weight_matrix_size1 = 128
depth1 = 3
weight_matrix_size2 = 128
depth2 = 32
fully_conn_layer = 256

img_size = 256
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 3
num_classes = 3


def convolutional_layer(input, depth, no_filters, weight_matrix_size):
    dimensions = [weight_matrix_size, weight_matrix_size, depth, no_filters]
    weights = weight_matrix(shape=dimensions)
    biases = biases_matrix(no_filters)
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases
    layer = tf.compat.v1.nn.max_pool(value=input, matrix_size=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

'''
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

'''
def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
'''
print('Training set', train_x.shape, train_y.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_x.shape, test_y.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden1 = 256
num_hidden2 = 64

graph = tf.Graph()

with graph.as_default():
    # Define the training dataset and lables
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    # Validation/test dataset
    # tf_valid_dataset = tf.constant(valid_dataset)
     # tf_test_dataset = tf.constant(test_dataset)

    # CNN layer 1 with filter (num_channels, depth) (3, 16)
    cnn1_W = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    cnn1_b = tf.Variable(tf.zeros([depth]))

    # CNN layer 2 with filter (depth, depth) (16, 16)
    cnn2_W = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    cnn2_b = tf.Variable(tf.constant(1.0, shape=[depth]))

    # Compute the output size of the CNN2 as a 1D array.
    size = image_size // 4 * image_size // 4 * depth

    # FC1 (size, num_hidden1) (size, 256)
    fc1_W = tf.Variable(tf.truncated_normal(
        [size, num_hidden1], stddev=np.sqrt(2.0 / size)))
    fc1_b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

    # FC2 (num_hidden1, num_hidden2) (size, 64)
    fc2_W = tf.Variable(tf.truncated_normal(
        [num_hidden1, num_hidden2], stddev=np.sqrt(2.0 / (num_hidden1))))
    fc2_b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

    # Classifier (num_hidden2, num_labels) (64, 10)
    classifier_W = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=np.sqrt(2.0 / (num_hidden2))))
    classifier_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        # First convolution layer with stride = 1 and pad the edge to make the output size the same.
        # Apply ReLU and a maximum 2x2 pool
        conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + cnn1_b)
        pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # Second convolution layer
        conv2 = tf.nn.conv2d(pool1, cnn2_W, [1, 1, 1, 1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + cnn2_b)
        pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        # Flattern the convolution output
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        # 2 FC hidden layers
        fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

        # Return the result of the classifier
        return tf.matmul(fc2, classifier_W) + classifier_b

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    # valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    # test_prediction = tf.nn.softmax(model(tf_test_dataset))

num_steps = 20001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_y.shape[0] - batch_size)
    batch_data = train_x[offset:(offset + batch_size), :, :, :]
    batch_labels = train_y[offset:(offset+batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      # print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))

# print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_y))

# Test accuracy: 95.2%
