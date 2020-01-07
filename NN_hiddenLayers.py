import tensorflow as tf
import pickle
import numpy as np

tf.compat.v1.disable_eager_execution()
print("EAGER EXECUTION DISABLED")
train_x = pickle.load(open("X_train.pickle", "rb"))
print("X", len(train_x))
train_y = pickle.load(open("y_train.pickle", "rb"))
print("y", len(train_y), "Shape", tf.shape(train_y))
test_x = pickle.load(open("X_test.pickle", "rb"))
print("X", len(test_x))
test_y = pickle.load(open("y_test.pickle", "rb"))
print("y", len(test_y))
print("DATA LOADED")
n_nodes_hl1 = 5000
n_nodes_hl2 = 5000
n_nodes_hl3 = 5000

n_classes = 3
batch_size = 10
hm_epochs = 3
print("BUILDING PLACEHOLDERS")
x = tf.compat.v1.placeholder('float')
y = tf.compat.v1.placeholder('float')
print("HIDDEN LAYER 1")
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.compat.v1.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1]))}
print("HIDDEN LAYER 2")
hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2]))}
print("HIDDEN LAYER 3")
hidden_3_layer = {'f_fum': n_nodes_hl3,
                  'weight': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3]))}
print("OUTPUT LAYER")
output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.compat.v1.random_normal([n_nodes_hl3, n_classes])),
                'bias': tf.Variable(tf.compat.v1.random_normal([n_classes])), }


# Nothing changes
def neural_network_model(data):

    print("Building Model")
    print("Data", data, "shape", tf.shape(data))
    print("W1", hidden_1_layer['weight'], "shape", tf.shape(hidden_1_layer['weight']))
    print("B1", hidden_1_layer['bias'], "shape", tf.shape(hidden_1_layer['bias']))
    l1 = tf.matmul(data, hidden_1_layer['weight'])
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden_2_layer['weight'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']

    return output


def train_neural_network(x):
    print("Training the model")
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(hm_epochs):
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

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)