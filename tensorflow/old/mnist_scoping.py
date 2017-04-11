'''
A Recurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Example code is adapted from https://github.com/aymericdamien/TensorFlow-Examples/
Author: Parminder
'''

import argparse
import tensorflow as tf
import numpy as np

from layers import *
import streaming

rnn_cell = tf.nn.rnn_cell

tf.app.flags.FLAGS = tf.python.platform.flags._FlagValues()
tf.app.flags._global_parser = argparse.ArgumentParser()

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/", one_hot=True)

'''
To classify images using a reccurent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("iterations", 100000, "Number of iterations.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("display_step", 10,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("hidden", 128,
                            "How many hidden units.")
tf.app.flags.DEFINE_integer("classes", 10,
                            "Number of classes")
tf.app.flags.DEFINE_integer("layers", 1,
                            "Number of layers for the model")
tf.app.flags.DEFINE_integer("dau", 1,
                            "Batches per update in for use in decoupled accumulation and update.")
tf.app.flags.DEFINE_string("cell_type", "SNGRU", "Select from LSTM, GRU , BasicRNN, LNGRU, LNLSTM, HyperLnLSTMCell")
tf.app.flags.DEFINE_boolean("hyper_layer_norm", False, "For HyperLnLSTMCell use only.")
# tf.app.flags.DEFINE_string("summaries_dir", "./log/", "Directory for summary")
FLAGS = tf.app.flags.FLAGS
# Parameters
learning_rate = FLAGS.learning_rate
training_iters = FLAGS.iterations
batch_size = FLAGS.batch_size
display_step = FLAGS.display_step

# Network Parameters
n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = FLAGS.hidden  # hidden layer num of features
n_classes = FLAGS.classes  # MNIST total classes (0-9 digits)

n_dau = FLAGS.dau

print FLAGS.dau

streaming_norm_training_mode_global_flag = False

def train():
    global streaming_norm_training_mode_global_flag
    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x-input')
        y = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases, type, hyper_layer_norm, scope=None):

        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        cell_class_map = {
            "LSTM": rnn_cell.BasicLSTMCell(n_hidden),
            "GRU": rnn_cell.GRUCell(n_hidden),
            "BasicRNN": rnn_cell.BasicRNNCell(n_hidden),
            "LNGRU": LNGRUCell(n_hidden),
            "SNGRU": SNGRUCell(n_hidden),
            "LNLSTM": LNBasicLSTMCell(n_hidden),
            'HyperLnLSTMCell': HyperLnLSTMCell(n_hidden, is_layer_norm=hyper_layer_norm)
        }

        lstm_cell = cell_class_map.get(type)
        cell = rnn_cell.MultiRNNCell([lstm_cell] * FLAGS.layers, state_is_tuple=True)
        print "Using %s model" % type
        # Get lstm cell output
        outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32, scope=scope)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    compute_grads = []
    pred = RNN(x_cur, weights, biases, FLAGS.cell_type, FLAGS.hyper_layer_norm)
    # Define loss and optimizer
    # print pred
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    grads = optimizer.compute_gradients(cost)
    grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
    # apply_placeholder_op = opt.apply_gradients(grad_placeholder)
    transform_grads = [(function1(grad[0]), grad[1]) for grad in grads]
    apply_transform_op = opt.apply_gradients(transform_grads)

    apply_grads = []
    for grad in compute_grads:
        apply_grads.append(optimizer.apply_gradients(grad))


    pred_single = RNN(x, weights, biases, FLAGS.cell_type, FLAGS.hyper_layer_norm)
    cost_single = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_single, y))
    correct_pred = tf.equal(tf.argmax(pred_single, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tf.scalar_summary('Accuracy', accuracy)
    # tf.scalar_summary('Cost', cost)

    # merged = tf.merge_all_summaries()
    # train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + "train/",
    #                                       sess.graph)
    # test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + "test/",
    #                                      sess.graph)

    # Initializing the variables
    init = tf.initialize_all_variables()
    for v in tf.trainable_variables():
        print v.name
    sess.run(init)
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    step = 1
    
    # Keep training until reach max iterations
    while step * batch_size * n_dau < training_iters:
        batches_x = []
        batches_y = []
        for i in range(n_dau):
            batch = mnist.train.next_batch(batch_size)
            batches_x.append(batch[0].reshape([batch_size, n_steps, n_input]))
            batches_y.append(batch[1])

        # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape([batch_size, n_steps, n_input])
        # Run optimization op (backprop)
        # summary, _ = sess.run([merged,optimizer], feed_dict={x: batch_x, y: batch_y})
        
        streaming_norm_training_mode_global_flag = True

        # summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
        # accumulate this
        #gradvar = optimizer.compute_gradients(cost)

        # ggvv = tf.gradients(cost, tf.trainable_variables()) # what is the difference?
        # not sure the answer is correct... but relevant
        # http://stackoverflow.com/questions/35226428/how-do-i-get-the-gradient-of-the-loss-at-a-tensorflow-variable
        sess.run(apply_grads, feed_dict={x_dau: batches_x, y_dau: batches_y})
        # train_writer.add_summary(summary, step)

        if step % display_step == 0:
            # Calculate batch accuracy
            # summary, acc, loss = sess.run([merged, accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            # train_writer.add_summary(summary, step)
            # Calculate batch loss
            acc, loss = sess.run([accuracy, cost_single], feed_dict={x: batches_x[-1], y: batches_y[-1]})
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc)
            streaming_norm_training_mode_global_flag = False
            # summary, acc, loss = sess.run([merged, accuracy, cost], feed_dict={x: test_data, y: test_label})
            # test_writer.add_summary(summary, step)
            acc, loss = sess.run([accuracy, cost_single], feed_dict={x: test_data, y: test_label})

            print "Testing Accuracy:", acc
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 128 mnist test images

    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label})


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
