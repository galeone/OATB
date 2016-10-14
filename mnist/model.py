# Copyright 2016 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Defines LeNet as showed into the Tensorflow MNIST tutorial"""
import tensorflow as tf

# define some handy functions in order to create variables easily


def weight_variable(shape):
    """Returns a new variable with the chosen shape, initialized with
    random values from a normal distribution"""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Returns a bias variabile initializeted to 0.1"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(input_x, W):
    """Convolve input_x with W"""
    return tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(input_x):
    """Max pooling 2x2"""
    return tf.nn.max_pool(
        input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def infer(input_x, keep_prob):
    """Returns the model"""
    # First convolutional layer: convolution follower by max pooling.
    # The conv layer will compute 32 features for each 5x5 patch. Its weight tensor will
    # have a shape of [5,5,1,32].
    # [5x5] = patch size
    # 1 = the number of input channel
    # 32 = the number of output channel
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # reshape x to a 4d tensors, with the second and third dim corresponding to image widht
    # and height. The final dimension corresponding to the number of color channels
    x_image = tf.reshape(input_x, [-1, 28, 28, 1])

    # convolve x_image with the weight tensor, add the bias, apply ReLU function and
    # finally max pool
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer: 64 features for each 5x5 patch
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer:
    #    now that the image size has been reduced to 7x7, we add a FC layer with 1024 neurons
    #    to allow processing on the entire image.
    #    We reshape the tensor from the pooling layer into a batch of vectors,
    #    multiply by a weight matrix, add a bias, and apply a ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout:
    #    to reduce overfitting we will apply dropout before the readout layer.
    #    Create a placeholder for the probability that a neuron's output is kept during
    #    dropout. This allow us to turn dropout on during training, and turn it off during
    #    testing. tf.nn.dropout automatically handles scaling nerons outputs in additins
    #    to making them. So dropout just works without any additianal scaling
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout layer:
    #    add a linear softmax layer (no non-linearity)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return logits


def loss(logits, labels):
    """ Return the loss function for the model returned by infer.
    Args:
        logits: Logits from infer()
        labels: one-hot encoded labels
    """

    return tf.reduce_mean(-tf.reduce_sum(
        labels * tf.log(tf.nn.softmax(logits)), reduction_indices=[1]))
