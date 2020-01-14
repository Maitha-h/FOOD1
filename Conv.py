import tensorflow.compat.v1 as tf
import pickle, time
import numpy as np
train_x = pickle.load(open("X_train.pickle", "rb"))
print("X", len(train_x))
train_y = pickle.load(open("y_train.pickle", "rb"))
print("y", len(train_y), "Shape", tf.shape(train_y))
test_x = pickle.load(open("X_test.pickle", "rb"))
print("X", len(test_x))
test_y = pickle.load(open("y_test.pickle", "rb"))
print("y", len(test_y))
print("DATA LOADED")
train_x = train_x / 255.0
n_classes = 3
batch_size = 32
print("Building placeholders")

x = tf.placeholder(tf.float32, [32, 256, 265, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
'''
x = tf.placeholder(tf.float32,1)
y = tf.placeholder(tf.float32,1)
'''
'''
x = tf.placeholder('float')
y = tf.placeholder('float')
'''

i = 0
j = 0

def conv(in_tensor, filters, weight_decay=0.0005, use_bn=True):
    global i
    input_size = in_tensor.get_shape().as_list()[3]
    shape = (3, 3, input_size, filters)
    l = tf.nn.l2_loss(tf.truncated_normal([256 * 256 * 3, n_classes]))
    w = tf.get_variable("W{}_".format(i + 1), shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(in_tensor, w, strides=(1,1,1,1), padding="VALID")
    i += 1
    return tf.nn.relu(conv)

def pcap(in_tensor, caps_dims):
    shape = in_tensor.get_shape().as_list()
    cap_count = shape[1]*shape[2]*shape[3]//caps_dims
    transposed = tf.transpose(in_tensor, [0,3,1,2]) # What is a perm
    return tf.reshape(transposed, [-1, 3, cap_count, caps_dims])


def hvc(in_tensor, out_caps, cap_dims, weight_decay=0.005):
    global j
    cap_size=in_tensor.get_shape().as_list()[2]
    # regularizer=weight_decay*tf.nn.l2_loss(tf.Variable(tf.truncated_normal([256 * 256 * 3, n_classes])))
    w_out_cap = tf.get_variable("w_out_cap{}_".format(j + 1),shape=[out_caps, cap_size, cap_dims], initializer=tf.glorot_uniform_initializer())
    ocap = tf.reduce_sum(tf.multiply(in_tensor, w_out_cap), 2)
    j += 1
    return tf.nn.relu(ocap)


def convolutional_neural_network(x):
    print(tf.shape(x))
    conv1 = conv(x, 32)
    conv2 = conv(conv1, 48)
    conv3 = conv(conv2, 64)
    cap_dims = conv3.get_shape().as_list()[3]*conv3.get_shape().as_list()[3]
    caps = pcap(conv3, cap_dims)
    ocap = hvc(caps, n_classes, cap_dims)
    logits = tf.reduce_sum(ocap, axis=2)
    preds = tf.nn.softmax(logits=logits)
    return preds


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
            print('Epoch', epoch + 1, 'completed out of', hm_epochs, '--------------------Loss:', epoch_loss,
                  'Accuracy:', accuracy.eval({x: test_x, y: test_y}))


Start_time = time.time()
train_neural_network(x)
print("EXECUTION TIME: ", int((time.time() - Start_time)//60), ":", int((time.time() - Start_time)%60))

# try it with eager execution


