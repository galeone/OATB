# Copyright 2016 Paolo Galeone <nessuno@nerdz.eu>. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, you can obtain one at http://mozilla.org/MPL/2.0/.
#Exhibit B is not attached; this software is compatible with the
#licenses expressed under Section 1.12 of the MPL v2.
"""Evaluates various optimization algorithms"""

import argparse
import os
import sys
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as mnist_data
from mnist.model import Model
from mnist import TRAIN_SIZE

BATCH_SIZE = 50
STEP_PER_EPOCH = TRAIN_SIZE / BATCH_SIZE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = CURRENT_DIR + "/log"


def main(args):
    """ Evaluates various optimization algorithms """

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # set graph level seed from random generators, in order to
        # get the same values for every random variable generated
        # for every model
        tf.set_random_seed(1)

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
        with tf.device(args.device):
            optimizers = [{
                "name": "Vanilla",
                "optimizer": tf.train.GradientDescentOptimizer(learning_rate),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/Vanilla'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "Momentum",
                "optimizer": tf.train.MomentumOptimizer(
                    learning_rate, momentum=0.5),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/Momentum'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "AdaGrad",
                "optimizer": tf.train.AdagradOptimizer(learning_rate),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/AdaGrad'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "AdaDelta",
                "optimizer": tf.train.AdadeltaOptimizer(),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/AdaDelta'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "RMSProp",
                "optimizer": tf.train.RMSPropOptimizer(learning_rate),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/RMSProp'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "ADAM",
                "optimizer": tf.train.AdamOptimizer(learning_rate),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/ADAM'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }, {
                "name": "FTRL",
                "optimizer": tf.train.FtrlOptimizer(learning_rate),
                "summary": tf.train.SummaryWriter(LOG_DIR + '/FTRL'),
                "logits": None,
                "loss": None,
                "train_step": None,
                "loss_summary": None,
            }]

            for obj in optimizers:
                # define the model
                model = Model(inputs_, keep_prob_)
                obj["logits"] = model.logits

                # define the loss
                obj["loss"] = model.loss(labels_)

                # define the train step
                obj["train_step"] = obj["optimizer"].minimize(obj["loss"])

                # log the loss value in the 'loss' summary
                obj["loss_summary"] = tf.scalar_summary('loss', obj["loss"])

            # define metrics in functions of placeholders
            predictions_ = tf.placeholder(tf.float32, shape=[None, 10])

            correct_prediction = tf.equal(
                tf.argmax(predictions_, 1), tf.argmax(labels_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # accuracy summary, using placeholder to define the name
        # It can be used to log train, validation and test accuracy
        accuracy_name_ = tf.placeholder(tf.string, shape=[])
        accuracy_summary = tf.scalar_summary(accuracy_name_, accuracy)

        init_op = tf.initialize_all_variables()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(init_op)

            # first batch, used to log initial loss value
            batch = datasets.train.next_batch(BATCH_SIZE)

            for obj in optimizers:
                # Log intial loss value, before training
                summary_line = sess.run(obj["loss_summary"],
                                        feed_dict={
                                            inputs_: batch[0],
                                            labels_: batch[1],
                                            keep_prob_: 1.0
                                        })
                obj["summary"].add_summary(summary_line, global_step=0)

            # loop for 2000 time, starting from 1 because of initial loss logging
            for i in range(1, 20001):
                batch = datasets.train.next_batch(BATCH_SIZE)
                for obj in optimizers:
                    _, summary_line, loss_value = sess.run(
                        [obj["train_step"], obj["loss_summary"], obj["loss"]],
                        feed_dict={
                            inputs_: batch[0],
                            labels_: batch[1],
                            keep_prob_: 0.5
                        })

                    obj["summary"].add_summary(summary_line, global_step=i)

                    if i % 100 == 0:
                        predictions_values = sess.run(
                            obj["logits"],
                            feed_dict={inputs_: batch[0],
                                       keep_prob_: 1.0})

                        train_accuracy, summary_line = sess.run(
                            [accuracy, accuracy_summary],
                            feed_dict={
                                labels_: batch[1],
                                predictions_: predictions_values,
                                accuracy_name_: "train_accuracy",
                            })

                        obj["summary"].add_summary(summary_line, global_step=i)
                        predictions_values = sess.run(
                            obj["logits"],
                            feed_dict={
                                inputs_: datasets.validation.images,
                                keep_prob_: 1.0
                            })

                        validation_accuracy, summary_line = sess.run(
                            [accuracy, accuracy_summary],
                            feed_dict={
                                labels_: datasets.validation.labels,
                                predictions_: predictions_values,
                                accuracy_name_: "validation_accuracy",
                            })

                        obj["summary"].add_summary(summary_line, global_step=i)

                        print(
                            "{}: step {}, training accuracy {} valdidation accuracy {}. Loss: {}".
                            format(obj["name"], i, train_accuracy,
                                   validation_accuracy, loss_value))

                    obj["summary"].flush()

            # End of train
            for obj in optimizers:
                predictions_values = sess.run(obj["logits"],
                                              feed_dict={
                                                  inputs_:
                                                  datasets.test.images,
                                                  keep_prob_: 1.0
                                              })

                test_accuracy, summary_line = sess.run(
                    [accuracy, accuracy_summary],
                    feed_dict={
                        labels_: datasets.test.labels,
                        predictions_: predictions_values,
                        accuracy_name_: "test_accuracy",
                    })
                obj["summary"].add_summary(summary_line)
                obj["summary"].flush()

                print(obj["name"], "test accuracy {}".format(test_accuracy))

        return 0


if __name__ == "__main__":
    ARG_PARSER = argparse.ArgumentParser(description="Evaluate the model")
    ARG_PARSER.add_argument("--device", default="/gpu:1")
    sys.exit(main(ARG_PARSER.parse_args()))
