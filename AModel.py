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



def weight_variable(shape):
    with tf.name_scope("weight"):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope("bias"):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def define_and_train():

    image_size = 256
    classes = 3
    input_layer_name = "input_tensor"
    output_layer_name = "softmax_tensor"

    train_dir = "tmp"
    learning_rate = 0.01
    batch_size = 32
    train_steps = 3000
    logging_step = 200
    checkpoint_step = 500

    # Define graph
    ############################################################

    input_layer = tf.placeholder(tf.float32, shape=[None, image_size * image_size], name=input_layer_name)
    input_image = tf.reshape(input_layer, shape=[-1, image_size, image_size, 1])

    # 1 Convolution layer
    conv1_w = weight_variable([5, 5, 1, 32])
    conv1_b = bias_variable([32])
    conv1 = tf.nn.relu(conv2d(input_image, conv1_w) + conv1_b)
    pool1 = max_pool_2x2(conv1)

    # 2 Convolution layer
    conv2_w = weight_variable([5, 5, 32, 64])
    conv2_b = bias_variable([64])
    conv2 = tf.nn.relu(conv2d(pool1, conv2_w) + conv2_b)
    pool2 = max_pool_2x2(conv2)

    # Flatten
    pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

    # 3 Fully connected layer
    full_layer1_w = weight_variable([8 * 8 * 64, 1024])
    full_layer1_b = bias_variable([1024])
    full_layer1 = tf.nn.relu(tf.matmul(pool2_flat, full_layer1_w) + full_layer1_b)

    # 4 Fully connected layer
    full_layer2_w = weight_variable([1024, classes])
    full_layer2_b = bias_variable([classes])
    full_layer2 = tf.matmul(full_layer1, full_layer2_w) + full_layer2_b

    # Output
    output = tf.nn.softmax(full_layer2, name=output_layer_name)  # softmax output
    pred = tf.argmax(output, axis=1)  # predictions

    # Placeholders used for training
    output_true = tf.placeholder(tf.float32, shape=[None, classes])
    pred_true = tf.argmax(output_true, axis=1)


    # Calculate loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=output_true, logits=full_layer2))
    # Configure training operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # Add evaluation metrics
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, pred_true), tf.float32))
    # Initialize variables (assign default values..)
    init = tf.global_variables_initializer()
    # Initialize saver
    saver = tf.train.Saver()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(init)
        summary_writer = tf.summary.FileWriter(train_dir, graph=tf.get_default_graph())

        for step in range(train_steps + 1):
            # Get random batch
            idx = np.random.randint(len(train_x), size=batch_size)
            batchX = train_x[idx, :]
            batchY = train_y[idx]

            # Run the optimizer
            _, train_loss, train_accuracy, summary = session.run(
                [optimizer, loss, accuracy, merged_summary_op],
                feed_dict={input_layer: batchX,
                           output_true: batchY}
            )
            # Add summary for tensorboard
            summary_writer.add_summary(summary, step)

            # Test training
            if step % logging_step == 0:
                test_loss, test_accuracy = session.run(
                    [loss, accuracy],
                    feed_dict={input_layer: test_x,
                               output_true: test_y}
                )

                print("Step {0:d}: Loss = {1:.4f}, Accuracy = {2:.3f}".format(step, test_loss, test_accuracy))


def main():
    define_and_train()


if __name__ == "__main__":
    main()