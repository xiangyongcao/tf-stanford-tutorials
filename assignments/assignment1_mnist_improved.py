# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:35:17 2017

@author: Xiangyong Cao
"""
## Fully connected neural network version (written by Xiangyong Cao)

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Process Data
mnist = input_data.read_data_sets('/data/mnist',one_hot=True)

# define parameters for the model
learning_rate = 0.01
batch_size = 100
n_epochs = 25
hidden1 = 256
hidden2 = 256

# Two Phases

## Phase 1: Assemble our graph

### step 1: Create placeholders for inputs and labels
X = tf.placeholder(tf.float32,shape=[None,784],name='image')
Y = tf.placeholder(tf.float32,shape=[None,10],name='label')
### step 2: Create weight and bias
W1 = tf.Variable(initial_value = tf.truncated_normal(shape=[784,hidden1],stddev=0.1),name='weights1')
b1 = tf.Variable(initial_value = tf.zeros([1,hidden1]),name='bias1')

W2 = tf.Variable(initial_value = tf.truncated_normal(shape=[hidden1,hidden2],stddev=0.1),name='weights2')
b2 = tf.Variable(initial_value = tf.zeros([1,hidden2]),name='bias2')

W3 = tf.Variable(initial_value = tf.truncated_normal(shape=[hidden2,10],stddev=0.1),name='weights3')
b3 = tf.Variable(initial_value = tf.zeros([1,10]),name='bias3')


### step 3: Bulid model
h_fc1 = tf.add(tf.matmul(X,W1),b1)
h_fc1 = tf.nn.relu(h_fc1)

h_fc2 = tf.add(tf.matmul(h_fc1,W2),b2)
h_fc2 = tf.nn.relu(h_fc2)

logits = tf.matmul(h_fc2,W3) + b3

### step 4: Specify loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits,Y)
loss = tf.reduce_mean(entropy)
### step 5: Create Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

## Phase 2: Train our model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    n_batches = int(mnist.train.num_examples/batch_size)
    
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            start_time = time.time()
            X_batch,Y_batch = mnist.train.next_batch(batch_size)
            _,loss_value,logits_batch = sess.run([optimizer,loss,logits],feed_dict={X:X_batch,Y:Y_batch})
#            preds = tf.nn.softmax(logits_batch)
#            correct_pred = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
#            num_correct_pred = tf.reduce_sum(tf.cast(correct_pred,tf.float32))
#            accuracy = sess.run(num_correct_pred)/batch_size
            duration = time.time() - start_time
#            print('Epoch %d (batch %d): loss = %.2f , accuracy = %.2f (%.3f sec)' 
#                  % (epoch, batch, loss_value, accuracy, duration))
            print('Epoch %d (batch %d): loss = %.2f (%.3f sec)' 
                  % (epoch, batch, loss_value, duration))
        preds = tf.nn.softmax(logits_batch)
        correct_pred = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
        num_correct_pred = tf.reduce_sum(tf.cast(correct_pred,tf.float32))
        accuracy = sess.run(num_correct_pred)/batch_size
        print('Training accuracy = %.2f' % (accuracy))
        
    ### test model
    n_batches = int(mnist.test.num_examples/batch_size)
    total_correct_preds = 0
    for _ in range(n_batches):
        X_batch,Y_batch = mnist.test.next_batch(batch_size)
        _,loss_batch,logits_batch = sess.run([optimizer,loss,logits],
                                             feed_dict={X:X_batch,Y:Y_batch})
        preds = tf.nn.softmax(logits_batch)
        correct_pred = tf.equal(tf.argmax(preds,1),tf.argmax(Y_batch,1))
        num_correct_pred = tf.reduce_sum(tf.cast(correct_pred,tf.float32))
        total_correct_preds += sess.run(num_correct_pred)
    print('---------------------------------------------------------------')
    print('Test Accuracy = %.2f' % (total_correct_preds/mnist.test.num_examples))
    











