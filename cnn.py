import parseData
import helpers
import layers
import os
import numpy as np
import tensorflow as tf

class Cnn():
    def __init__(self, params): 
        self.aminos = parseData.getAminos()
        self.allPops, self.allTurnCombs = parseData.getSeqInfo()
        self.allTurnCombs = self.allTurnCombs.keys()

        self.beta = params.regBeta
        self.numHiddenNodes = params.numHiddenNodes
        self.numX = params.numX
        

        self.batchSize = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32,shape=[params.numCLayers+1])
        self.X = tf.placeholder(tf.float32, shape=(None, self.numX + 1,len(self.aminos)), name="X")
        self.y = tf.placeholder(tf.float32, shape=(None, len(self.allTurnCombs)), name="y")

        self.logits = -1
        self.optimizer = -1
        self.loss = -1

        self.setUpArch(self.X, params.numChannels, params.numCLayers,\
                                params.filterHeights, params.leakSlope)

    def setUpArch(self, nnInput, numChannels, numCLayers, filterHeights, leakSlope):
        cnnOut = self.cnn(self.X, numChannels, numCLayers, filterHeights, leakSlope)
        fcWordDim = (len(filterHeights) * numChannels[-1] + len(self.aminos))\
                            * cnnOut.get_shape()[1]
        fcInput = tf.reshape(tf.concat(axis=2, values=[cnnOut, self.X]), \
                                                                [-1,fcWordDim])

        if self.numHiddenNodes != 0:
            fcOut = layers.linear([fcInput], self.numHiddenNodes, True, 1.0, scope="fc")
            sx_inputs = tf.nn.relu(fcOut)
            sx_inputs = tf.nn.dropout(sx_inputs,rate=1-self.keep_prob[-1])
            self.logits = layers.linear([sx_inputs], len(self.allTurnCombs), True, 1.0, scope="softmax")
        else:
            self.logits = layers.linear([fcInput], len(self.allTurnCombs), True, 1.0, scope="softmax")

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
        self.loss=tf.reduce_mean(xentropy, name="loss") 

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                   
    def cnn(self, nnInput, numChannels, numCLayers, filterHeights, leakSlope):
        layerIn = nnInput
        imgH = layerIn.get_shape()[1].value
        with tf.variable_scope("cnn_layer{0}".format(0)):
            imgW = layerIn.get_shape()[2].value
            cnnFilterIn = tf.reshape(layerIn, [-1, imgH, imgW, 1])
            layerOut = layers.multiChannelCnn(cnnFilterIn, imgH, imgW, \
                                        filterHeights, self.batchSize,\
                                        numChannels[0], leakSlope)
            #layerOut = tf.layers.batch_normalization(layerOut, training=True)
            layerIn = layerOut
            layerIn = tf.nn.dropout(layerIn, rate=1-self.keep_prob[0])
        for i in range(1, numCLayers):
            inputDim = layerIn.get_shape()[2].value
            layerInB = tf.reshape(layerIn, [-1, inputDim])
            with tf.variable_scope("cnn_gates"):
                if i > 1:
                    tf.get_variable_scope().reuse_variables()
                u = layers.linear([layerInB], inputDim, True, 1.0, scope="gate")
                u = tf.sigmoid(u)
            input_shape = layerIn.get_shape().as_list()
            
            with tf.variable_scope("cnn_layer{0}".format(i)):
                imgW = layerIn.get_shape()[2].value
                cnnLayerIn = tf.reshape(layerIn, [-1, imgH, imgW, 1])
                layerOut = layers.multiChannelCnn(cnnLayerIn, imgH, imgW,\
                                filterHeights, self.batchSize, numChannels[i],leakSlope)
                layerOutputB = tf.reshape(layerOut, [-1, inputDim])
                layerOut = u * layerInB + (1-u) * layerOutputB
                input_shape[0] = -1
                layerOut = tf.reshape(layerOut, input_shape)
                layerIn = layerOut
                layerIn = tf.nn.dropout(layerIn, rate=1-self.keep_prob[i])
        cnnOut = layerIn
        return cnnOut

    def createDict(self, keep_prob, X, batchSize, y=None, learning_rate=None):
        fdict = {}
        fdict[self.keep_prob] = keep_prob
        fdict[self.X] = X
        fdict[self.batchSize] = batchSize
        if y is not None:
            fdict[self.y] = y
        if learning_rate is not None:
            fdict[self.learning_rate] = learning_rate
        return fdict

    def getLoss(self):
        return self.loss

    def getOptimizer(self):
        return self.optimizer

    def getLogits(self):
        return self.logits

    def set_lr(self, new_lr):
        self.learning_rate = new_lr

    def get_lr(self):
        return self.learning_rate
