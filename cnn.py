#Aidan Fike
#June 11, 2019

#Class implementing a convolutional neural network

import parseData
import helpers
import layers
import os
import numpy as np
import tensorflow as tf

class Cnn():
    def __init__(self, params): 
        #Information about the type of data that it will recieve: turn
        #combinations and aminos acids
        self.aminos = parseData.getAminos()
        self.allPops, self.allTurnCombs = parseData.getSeqInfo()
        self.allTurnCombs = self.allTurnCombs.keys()

        #Various relevant hyperparameters, regularization, number of nodes, and
        #number of amino acids given in the representation. In this case, this
        #will be numX + numExtraX amino acids with one-hot encoding
        self.beta = params.regBeta
        self.numHiddenNodes = params.numHiddenNodes
        self.numX = params.numX
        self.numExtraX = params.numExtraX
        

        #More hyperparameters that will be passed in at runtime
        self.batchSize = tf.placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32,shape=[params.numCLayers+1])

        #Instances and relevant labels
        self.X = tf.placeholder(tf.float32, shape=(None, self.numX + self.numExtraX,\
                                                    len(self.aminos)), name="X")
        self.y = tf.placeholder(tf.float32, shape=(None, len(self.allTurnCombs)), name="y")

        #Operations for the logits, optimizer, and loss functions
        self.logits = -1
        self.optimizer = -1
        self.loss = -1

        #Create the model itself
        self.setUpArch(self.X, params.numChannels, params.numCLayers,\
                                params.filterHeights, params.leakSlope)

    #Set up the architecture for the convolutional neural network
    #
    #Params: numChannels - an array containing the number of channels at each
    #                      layer
    #        numCLayers - the number of convolutional layers. Integer
    #        filterHeights - the various heights of the filters. Array of ints
    #        leakSlope - the slope of the leaky relu
    def setUpArch(self, numChannels, numCLayers, filterHeights, leakSlope):

        #Create a cnn, gaining the last layer of the cnn as a return
        cnnOut = self.cnn(self.X, numChannels, numCLayers, filterHeights, leakSlope)
    
        #Parameters for the fully connected layer
        fcWordDim = (len(filterHeights) * numChannels[-1] + len(self.aminos))\
                            * cnnOut.get_shape()[1]
        fcInput = tf.reshape(tf.concat(axis=2, values=[cnnOut, self.X]), \
                                                                [-1,fcWordDim])
        
        #If there is a fully connected hidden layer, create one and then create
        #the logits (the values before the softmax function). Otherwise, just
        #create the logits
        if self.numHiddenNodes != 0:
            fcOut = layers.linear([fcInput], self.numHiddenNodes, True, 1.0, scope="fc")
            sx_inputs = tf.nn.relu(fcOut)
            sx_inputs = tf.nn.dropout(sx_inputs,rate=1-self.keep_prob[-1])
            self.logits = layers.linear([sx_inputs], len(self.allTurnCombs), True, 1.0, scope="softmax")
        else:
            self.logits = layers.linear([fcInput], len(self.allTurnCombs), True, 1.0, scope="softmax")

        #The loss function is cross softmax entropy with the logits created
        #above and the labels passed in above
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
        self.loss=tf.reduce_mean(xentropy, name="loss") 

        #Create an optimizer to minimize the loss function
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                   
    #Create a convolutional neural network using the various passed parameters 
    #
    #Params: nnInput - the instances that are passed into the network. 3D array 
    #                  of floats: first dimension is number of instances,
    #                  second is number of amino acids, 3rd is amino acid choices
    #        numChannels - the number of each type filter for each layer. 
    #                      an array with this number for each convolutional
    #                      layer 
    #        numCLayers - the number of convolutional layers
    #        filterHeights - the number of different heights for the filters
    #        leakSlop - the slope for the leaky relu function
    #
    # Return - the final layer of the network
    def cnn(self, nnInput, numChannels, numCLayers, filterHeights, leakSlope):
        layerIn = nnInput
        imgH = layerIn.get_shape()[1].value
        #Create the first layer of the convolutional neural network
        with tf.variable_scope("cnn_layer{0}".format(0)):
            imgW = layerIn.get_shape()[2].value
            cnnFilterIn = tf.reshape(layerIn, [-1, imgH, imgW, 1])
            layerOut = layers.multiChannelCnn(cnnFilterIn, imgH, imgW, \
                                        filterHeights, self.batchSize,\
                                        numChannels[0], leakSlope)
            #layerOut = tf.layers.batch_normalization(layerOut, training=True)
            layerIn = layerOut

            #Apply dropout to the outputted layer, using the keep_prob hyperparameter
            layerIn = tf.nn.dropout(layerIn, rate=1-self.keep_prob[0])

        #Create the rest of the layers in thre network
        for i in range(1, numCLayers):

            #Add the highway, to pass in the previous output to the output of 
            #the current layer. Measure the amount that should be passed with 
            #the variable u, an series of values created by the output of a
            #linear layer
            inputDim = layerIn.get_shape()[2].value
            layerInB = tf.reshape(layerIn, r[-1, inputDim])
            with tf.variable_scope("cnn_gates"):
                if i > 1:
                    tf.get_variable_scope().reuse_variables()
                u = layers.linear([layerInB], inputDim, True, 1.0, scope="gate")
                u = tf.sigmoid(u)
            input_shape = layerIn.get_shape().as_list()
            
            #Create another convolutional layer, taking the output of the last
            #as the input for the current. Then, apply the highway, adding
            #together some of the current layer with some of the previous
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

        #Return the final layer
        return cnnOut

    #Create a dictionary that can be used to feed in values into this network.
    #The values that can be fed in are fairly self explainatory
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
