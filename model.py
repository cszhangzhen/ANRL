# Representation Learning Class

import numpy as np
import tensorflow as tf
import time
import random
import math


class Model:
    def __init__(self, config, N, dims, X_target):
        self.config = config
        self.N = N
        self.dims = dims

        self.labels = tf.placeholder(tf.int32, shape=[None, 1])
        self.inputs = tf.placeholder(tf.int32, shape=[None])
        self.X_target = tf.constant(X_target, dtype=tf.float32)
        self.X_new = tf.nn.embedding_lookup(self.X_target, self.inputs)

        ############ define variables for autoencoder ##################
        self.layers = len(config.struct)
        self.struct = config.struct
        self.W = {}
        self.b = {}
        struct = self.struct

        # encode module
        for i in range(self.layers - 1):
            name_W = "encoder_W_" + str(i)
            name_b = "encoder_b_" + str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [struct[i], struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [struct[i+1]], initializer=tf.zeros_initializer())

        # decode module
        struct.reverse()
        for i in range(self.layers - 1):
            name_W = "decoder_W_" + str(i)
            name_b = "decoder_b_" + str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [struct[i], struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [struct[i+1]], initializer=tf.zeros_initializer())
        self.struct.reverse()

        ############## define input ###################
        self.X = tf.placeholder(tf.float32, shape=[None, config.struct[0]])

        self.make_compute_graph()
        self.make_autoencoder_loss()

        # compute gradients for deep autoencoder
        self.train_opt_ae = tf.train.AdamOptimizer(config.ae_learning_rate).minimize(self.loss_ae)

        ############ define variables for skipgram  ####################
        # construct variables for nce loss
        self.nce_weights = tf.get_variable("nce_weights", [
                                           self.N, self.dims], initializer=tf.contrib.layers.xavier_initializer())
        self.nce_biases = tf.get_variable(
            "nce_biases", [self.N], initializer=tf.zeros_initializer())

        self.loss_sg = self.make_skipgram_loss()

        # compute gradients for skipgram
        self.train_opt_sg = tf.train.AdamOptimizer(config.sg_learning_rate).minimize(self.loss_sg)

    def make_skipgram_loss(self):
        loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(
            weights=self.nce_weights,
            biases=self.nce_biases,
            labels=self.labels,
            inputs=self.Y,
            num_sampled=self.config.num_sampled,
            num_classes=self.N))

        return loss

    def make_compute_graph(self):

        def encoder(X):
            for i in range(self.layers - 1):
                name_W = "encoder_W_" + str(i)
                name_b = "encoder_b_" + str(i)
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
            return X

        def decoder(X):
            for i in range(self.layers - 1):
                name_W = "decoder_W_" + str(i)
                name_b = "decoder_b_" + str(i)
                X = tf.nn.tanh(tf.matmul(X, self.W[name_W]) + self.b[name_b])
            return X

        self.Y = encoder(self.X)

        self.X_reconstruct = decoder(self.Y)

    def make_autoencoder_loss(self):

        def get_autoencoder_loss(X, newX):
            return tf.reduce_sum(tf.pow((newX - X), 2))

        def get_reg_loss(weights, biases):
            reg = tf.add_n([tf.nn.l2_loss(w) for w in weights.itervalues()])
            reg += tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return reg

        loss_autoencoder = get_autoencoder_loss(self.X_new, self.X_reconstruct)

        loss_reg = get_reg_loss(self.W, self.b)

        self.loss_ae = self.config.alpha * loss_autoencoder + self.config.reg * loss_reg
