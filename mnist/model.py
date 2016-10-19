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
import uuid
import tensorflow as tf
from . import utils


class Model:
    def __init__(self, input_x, keep_prob):
        # associate a UUID with the object
        self._uuid = str(uuid.uuid1())
        # prefix every variable in the model with the uuid
        self.logits = self._infer(input_x, keep_prob)

    def _infer(self, input_x, keep_prob):
        """Returns the model"""
        x_image = tf.reshape(input_x, [-1, 28, 28, 1])

        with tf.variable_scope(self._uuid + "conv1"):
            conv1 = tf.nn.relu(
                utils.conv_layer(x_image, [5, 5, 1, 32], 1, 'SAME'))

        with tf.variable_scope(self._uuid + "pool1"):
            pool1 = tf.nn.max_pool(
                conv1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID')

        with tf.variable_scope(self._uuid + "conv2"):
            conv2 = tf.nn.relu(
                utils.conv_layer(pool1, [5, 5, 32, 64], 1, 'SAME'))

        with tf.variable_scope(self._uuid + "pool2"):
            pool2 = tf.nn.max_pool(
                conv2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID')

        with tf.variable_scope(self._uuid + "fc1"):
            W_fc1 = utils.weight("W", [7 * 7 * 64, 1024])
            b_fc1 = utils.bias("b", [1024])

            pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
            fc1 = tf.nn.relu(
                tf.nn.bias_add(tf.matmul(pool2_flat, W_fc1), b_fc1))
            fc1 = tf.nn.dropout(fc1, keep_prob)

        with tf.variable_scope(self._uuid + "softmax_linear"):
            W_fc2 = utils.weight("W", [1024, 10])
            b_fc2 = utils.bias("b", [10])
            logits = tf.nn.bias_add(tf.matmul(fc1, W_fc2), b_fc2, name="out")
            return logits

    def loss(self, labels):
        """ Return the loss function for the model returned by infer.
        Args:
            logits: Logits from infer()
            labels: one-hot encoded labels
        """
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.logits, labels))

    def variables(self):
        """Returns trainable model variables"""
        return [
            variable for variable in tf.trainable_variables()
            if variable.name.startswith(self._uuid)
        ]
