# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:26:53 2017

@author: Xiangyong Cao
"""

## Fully connected neural network version (written by Xiangyong Cao)

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

# training parameters
learning_rate = 0.01
batch_size = 100
n_epochs = 25

# network parameters
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

# Store layers weight & bias

with tf.name_scope('weight'):
    normal_weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='w1_normal'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='w2_normal'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name='wout_normal')
    }
    truncated_normal_weights  = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1),name='w1_truncated_normal'),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1),name='w2_truncated_normal'),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],stddev=0.1),name='wout_truncated_normal')
    }
    xavier_weights  = {
        'h1': tf.get_variable('w1_xaiver', [n_input, n_hidden_1],initializer=tf.contrib.layers.xavier_initializer()),
        'h2': tf.get_variable('w2_xaiver', [n_hidden_1, n_hidden_2],initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('wout_xaiver',[n_hidden_2, n_classes],initializer=tf.contrib.layers.xavier_initializer())
    }
    he_weights = {
        'h1': tf.get_variable('w1_he', [n_input, n_hidden_1],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'h2': tf.get_variable('w2_he', [n_hidden_1, n_hidden_2],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'out': tf.get_variable('wout_he', [n_hidden_2, n_classes],
                               initializer=tf.contrib.layers.variance_scaling_initializer())
    }
with tf.name_scope('bias'):
    normal_biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='b1_normal'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]),name='b2_normal'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='bout_normal')
    }
    zero_biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]),name='b1_zero'),
        'b2': tf.Variable(tf.zeros([n_hidden_2]),name='b2_zero'),
        'out': tf.Variable(tf.zeros([n_classes]),name='bout_normal')
    }
weight_initializer = {'normal':normal_weights, 'truncated_normal':truncated_normal_weights, 
                      'xavier':xavier_weights, 'he':he_weights}
bias_initializer = {'normal':normal_biases, 'zero':zero_biases}


# user input
from argparse import ArgumentParser
def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weight-init',
                        dest='weight_initializer', help='weight initializer',
                        metavar='WEIGHT_INIT', required=True)
    parser.add_argument('--bias-init',
                        dest='bias_initializer', help='bias initializer',
                        metavar='BIAS_INIT', required=True)
    parser.add_argument('--batch-norm',
                        dest='batch_normalization', help='boolean for activation of batch normalization',
                        metavar='BACH_NORM', required=True)
    return parser

# Batch normalization implementation
# from https://github.com/tensorflow/tensorflow/issues/1122
def batch_norm_layer(inputT, is_training=True, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                    lambda: batch_norm(inputT, is_training=True,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9, scope=scope),
                    lambda: batch_norm(inputT, is_training=False,
                    center=True, scale=True, activation_fn=tf.nn.relu, decay=0.9,
                    scope=scope, reuse = True))

# Create model of MLP with batch-normalization layer
def MLPwithBN(x, weights, biases, is_training=True):
    with tf.name_scope('MLPwithBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = batch_norm_layer(layer_1,is_training=is_training, scope='layer_1_bn')
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = batch_norm_layer(layer_2, is_training=is_training, scope='layer_2_bn')
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Create model of MLP without batch-normalization layer
def MLPwoBN(x, weights, biases):
    with tf.name_scope('MLPwoBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer                    
                    

def main():
    parser = build_parser()
    options = parser.parse_args()

    # Import Data
    mnist = input_data.read_data_sets('/data/mnist',one_hot=True)
    
    ## Phase 1: Assemble our graph
    ### step 1: Create placeholders for inputs and labels
    X = tf.placeholder(tf.float32,shape=[None,784],name='image')
    Y = tf.placeholder(tf.float32,shape=[None,10],name='label')
    ### placeholder for whether training batch_normalization parameters
    is_training = tf.placeholder(tf.bool,name='mode')
    
    ### step 2: create weights and bias
    weights = weight_initializer[options.weight_initializer]
    biases = bias_initializer[options.bias_initializer]
    batch_normalization = options.batch_normalization
    
    ### step 3: Build model
    if batch_normalization == 'True':
        logits = MLPwithBN(X, weights, biases, is_training=True)
    else:
        logits = MLPwoBN(X, weights, biases)
        
    ### step 4: Specify loss function
    with tf.name_scope('LOSS'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,Y))
    
    ### step 5: Create Optimizer
    with tf.name_scope('OPTI'):
        train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    
    ### moving_mean and moving_variance need to be updated
    if batch_normalization == "True":
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # don't understand
        if update_ops:
            train_ops = [train_step] + update_ops
            train_op_final = tf.group(*train_ops)
        else:
            train_op_final = train_step
    else:
        train_op_final = train_step
            
    ### calculate model accuracy 
    with tf.name_scope('ACC'):
        correct_predictions = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
        
    ### Create Summary
    tf.scalar_summary('loss',loss)
    tf.scalar_summary('acc',accuracy)
    # Merge all summaries into a single op
    merged_summary_op = tf.merge_all_summaries()

    ## Phase 2: Train our model
    init = tf.global_variables_initializer()
    
    with tf.InteractiveSession() as sess:
        sess.run(init,feed_dict={is_training: True})
        
        n_batches = int(mnist.train.num_examples/batch_size)
        
        # op to write logs to Tensorboard
        summary_writer = tf.train.SummaryWriter("logs/train", graph=tf.get_default_graph())
        
        for epoch in range(n_epochs):
            
            for batch in range(n_batches):
                
                start_time = time.time()
                
                X_batch,Y_batch = mnist.train.next_batch(batch_size)
                
                _,loss_value,train_accuracy,summary = sess.run([train_op_final,loss,accuracy,merged_summary_op],
                                                    feed_dict={X: X_batch,Y: Y_batch,is_training: True})
                
                duration = time.time() - start_time
                
                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * n_batches + batch)
            
                # Display logs
                if batch % 50 == 0:
                    print("Epoch:", '%04d,' % (epoch + 1),
                          "batch_index %4d/%4d, loss %.2f, training accuracy %.2f (%.2f sec)" 
                          % (batch, n_batches, loss_value, train_accuracy, duration))

        # Calculate accuracy for all mnist test images
        print("Test accuracy: %.2f" % accuracy.eval(
            feed_dict={X: mnist.test.images, Y: mnist.test.labels, is_training: False}))

if __name__ == '__main__':
    main()
        
    











