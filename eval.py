# Copyright 2016 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Evaluates various optimization algorithms, using LeNet and the MNIST dataset"""

import sys
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist_data
from mnist import model


def main():
    """ Evaluates various optimization algorithms, using LeNet and the MNIST dataset"""

    with tf.device('/cpu:0'):
        # get the data
        datasets = mnist_data.input_data.read_data_sets(
            'MNIST_data', one_hot=True)

        # define placeholders to feed the model
        # create nodes for the input images and target output classes
        # placeholders are values that we input when we ask tf to run a computation
        inputs_ = tf.placeholder(tf.float32, shape=[None, 784])
        labels_ = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob_ = tf.placeholder(tf.float32)

    # define the optimizers
    learning_rate = 1e-4

    with tf.device("/gpu:1"):
        optimizers = [{
            "name": "Vanilla",
            "optimizer": tf.train.GradientDescentOptimizer(learning_rate),
            "summary": tf.train.SummaryWriter('log/Vanilla'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "Momentum",
            "optimizer": tf.train.MomentumOptimizer(
                learning_rate, momentum=0.5),
            "summary": tf.train.SummaryWriter('log/Momentum'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "AdaGrad",
            "optimizer": tf.train.AdagradOptimizer(learning_rate),
            "summary": tf.train.SummaryWriter('log/AdaGrad'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "AdaDelta",
            "optimizer": tf.train.AdadeltaOptimizer(),
            "summary": tf.train.SummaryWriter('log/AdaDelta'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "RMSProp",
            "optimizer": tf.train.RMSPropOptimizer(learning_rate),
            "summary": tf.train.SummaryWriter('log/RMSProp'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "ADAM",
            "optimizer": tf.train.AdamOptimizer(learning_rate),
            "summary": tf.train.SummaryWriter('log/ADAM'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }, {
            "name": "FTRL",
            "optimizer": tf.train.FtrlOptimizer(learning_rate),
            "summary": tf.train.SummaryWriter('log/FTRL'),
            "logits": None,
            "loss": None,
            "train_step": None,
        }]

        for obj in optimizers:
            # define the model
            obj["logits"] = model.infer(inputs_, keep_prob_)

            # define the loss
            obj["loss"] = model.loss(obj["logits"], labels_)

            # define the train step
            obj["train_step"] = obj["optimizer"].minimize(obj["loss"])

            with tf.device('/cpu:0'):
                # log the loss value in the 'loss' summary
                loss_summary = tf.scalar_summary('loss', obj["loss"])

        # define metrics in functions of placeholders
        predictions_ = tf.placeholder(tf.float32, shape=[None, 10])

        correct_prediction = tf.equal(
            tf.argmax(predictions_, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.device('/cpu:0'):
        # accuracy summary, using placeholder to define the name
        # It can be used to log train and test accuracy
        accuracy_name_ = tf.placeholder(tf.string, shape=[])
        accuracy_summary = tf.scalar_summary(accuracy_name_, accuracy)

        init_op = tf.initialize_all_variables()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        for i in range(20000):
            batch = datasets.train.next_batch(50)

            for obj in optimizers:
                _, summary_line = sess.run([obj["train_step"], loss_summary],
                                           feed_dict={
                                               inputs_: batch[0],
                                               labels_: batch[1],
                                               keep_prob_: 0.5
                                           })

                obj["summary"].add_summary(summary_line, global_step=i)

                if i % 100 == 0:
                    predictions_values = sess.run(obj["logits"],
                                                  feed_dict={
                                                      inputs_: batch[0],
                                                      labels_: batch[1],
                                                      keep_prob_: 1.0
                                                  })

                    train_accuracy, summary_line = sess.run(
                        [accuracy, accuracy_summary],
                        feed_dict={
                            inputs_: batch[0],
                            labels_: batch[1],
                            keep_prob_: 1.0,
                            predictions_: predictions_values,
                            accuracy_name_: "train_accuracy",
                        })

                    obj["summary"].add_summary(summary_line, global_step=i)
                    print("step {}, training accuracy {}".format(
                        i, train_accuracy))

                obj["summary"].flush()

        for obj in optimizers:
            predictions_values = sess.run(obj["logits"],
                                          feed_dict={
                                              inputs_: datasets.test.images,
                                              labels_: datasets.test.labels,
                                              keep_prob_: 1.0
                                          })

            test_accuracy, summary_line = sess.run(
                [accuracy, accuracy_summary],
                feed_dict={
                    inputs_: datasets.test.images,
                    labels_: datasets.test.labels,
                    keep_prob_: 1.0,
                    predictions_: predictions_values,
                    accuracy_name_: "test_accuracy",
                })
            obj["summary"].add_summary(summary_line)
            obj["summary"].flush()

            print(obj["name"], "test accuracy {}".format(test_accuracy))

        return 0


if __name__ == "__main__":
    sys.exit(main())
