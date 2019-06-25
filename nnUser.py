#Aidan Fike
#June 12, 2019

#Implement a class to save training and testing instances, train a
#convolutional neural network, and make predictions with it

import parseData
import helpers
import os
import sys
import shutil
import math
import numpy as np
import tensorflow as tf

class nnUser():
    def __init__(self, params): 
        #Information about the instances the cnn uses
        self.allPops, self.allTurnCombs = parseData.getSeqInfo()
        self.allTurnCombs = self.allTurnCombs.keys()
        self.energy = parseData.getEnergyInfo()

        self.aminos = parseData.getAminos()
        self.numX = params.numX
        self.numExtraX = params.numExtraX

        #Whether to print the cost over time to command line
        self.verbose = params.verbose

        #Lists for the training and testing information (instances, labels, and
        #sequences) 
        self.trainInsts = []
        self.trainLabels = []
        self.testInsts = []
        self.testLabels = []
        self.testSeqs = []
        self.trainSeqs = []
        self.energyTrainVec = []
        self.energyTestVec = []

        self.params = params
        
        #Import hyperparameters
        self.metric = params.metric
        if self.metric != "pearson" and self.metric != "md" and \
                            self.metric != "rmsd" and self.metric != "crossEnt":
            print self.metric, "Not yet implemented. Add it to the predict function"

        self.calcMetStep = params.calcMetStep
        self.lr_decay = params.lr_decay
        self.lr = params.learning_rate
        self.lrThresh = params.lrThresh
        self.largeDecay = params.largeDecay
        self.keep_prob = params.keep_prob
        self.n_epochs = params.n_epochs
        self.decay_epoch = params.decay_epoch
        self.nImproveTime = params.nImproveTime

    #Make the training and testing instances blank
    def resetTrainTest(self):
        self.testSeqs = []
        self.trainSeqs = []

        self.trainInsts = []
        self.trainLabels = []
        self.testInsts = []
        self.testLabels = []

        self.energyTrainVec = []
        self.energyTestVec = []

    #Fill the training and testing information with based on the passed in test
    #sequences and the users goal (dataType)
    def genData(self, testSeqs, dataType):
        self.resetTrainTest()

        #Generate the training and testing sequences
        self.testSeqs = testSeqs
        self.trainSeqs = helpers.createTrainSeqs(self.allPops, testSeqs)

        turnCount = 0

        #Go through every known structural ensemble and add it to either the
        #training or testing set
        for turnComb in self.allTurnCombs:
            self.trainLabels.append([])
            self.testLabels.append([])
            currTurnDict = parseData.getTurnCombPops(self.allPops, turnComb)
            for seq in currTurnDict.keys():
                if not helpers.inSeqs(seq, testSeqs) or dataType == "train":
                    self.trainLabels[turnCount].append(currTurnDict[seq] / 100.0)
                elif seq in testSeqs:
                    self.testLabels[turnCount].append(currTurnDict[seq] / 100.0)

                if turnCount == 0:
                    seqVec = self.genSeqVec2D(seq, self.numX + self.numExtraX)
                    if not helpers.inSeqs(seq, testSeqs) or dataType == "train":
                        self.trainInsts.append(seqVec)
                    elif seq in testSeqs:
                        self.testInsts.append(seqVec)

            turnCount+=1

        if self.params.energyOn:
            testCount = 0
            trainCount = 0
            for seq in self.allPops:
                if not helpers.inSeqs(seq, testSeqs) or dataType == "train":
                    self.energyTrainVec.append([])
                    for turnComb in self.allTurnCombs:
                        if turnComb != "other":
                            currEnVec = self.genEnergyVec(seq,turnComb)
                            for energyTerm in currEnVec:
                                if energyTerm != 0.0:
                                    self.energyTrainVec[trainCount].append(energyTerm)
                    trainCount += 1

                elif seq in testSeqs:
                    self.energyTestVec.append([])
                    for turnComb in self.allTurnCombs:
                        if turnComb != "other":
                            currEnVec = self.genEnergyVec(seq,turnComb)
                            for energyTerm in currEnVec:
                                if energyTerm != 0.0:
                                    self.energyTestVec[testCount].append(energyTerm)
                    testCount += 1

        #Convert the training and testing information into numpy arrays
        self.energyTrainVec = np.array(self.energyTrainVec)
        self.energyTestVec = np.array(self.energyTestVec)
        self.energyTrainVec = np.array(self.energyTrainVec)
        self.energyTestVec = np.array(self.energyTestVec)
        self.trainLabels = np.array(self.trainLabels).transpose()
        self.testLabels = np.array(self.testLabels).transpose()
        self.trainInsts = np.array(self.trainInsts)
        self.testInsts = np.array(self.testInsts)
        
    #Generate a 2D representation of a passed sequence. Each row is a one-hot
    #encoding of each amino acid in the sequence. If vecLength > len(seq),
    #the first amino acids are added to the end of the generated array
    def genSeqVec2D(self, seq, vecLength):
        inv_aminos_map = {v: k for k, v in self.aminos.iteritems()} 
        seq_vec = np.zeros((vecLength, len(self.aminos)))
        for index, amino in enumerate(seq):
            if (index < self.numX):
                seq_vec[index, inv_aminos_map[amino]] = 1
        for index, amino in enumerate(seq):
            if (index < vecLength - self.numX):
                seq_vec[index + self.numX, inv_aminos_map[amino]] = 1

        return seq_vec 
   
    #Generate a vector used to represent the energetics of a given sequence
    #taking a given turn combination
    def genEnergyVec(self, seq, turnComb):
        energyVec = np.zeros(len(self.energy["AAADSV"]["II_0II_3"]))
        if turnComb in self.energy[seq]:
            for index, energy in enumerate(self.energy[seq][turnComb]):
                energyVec[index] = self.energy[seq][turnComb][energy]
        else:
            for index, energy in enumerate(self.energy[seq][turnComb]):
                energyVec[index] = -3.0
                #if energy == "PE_W" or energy == "PE_P" or energy == "PE_PW":

        return energyVec


    #Train passed in neural network
    #
    #Params: sess - the tensorflow session used to run the optimization
    #        model - the neural network being trained
    #        outputCost - a boolean of whether or not to output the cost to a
    #        file
    #
    # Return: A list of times and costs of the training process, lists of
    # training and testing accuracy through the training. This will be
    # outputted less than cost. The best testing metric found in the training
    # process
    def trainNet(self, sess, model, testNum, production, saveBest, outputCost):
        costs = []
        metTest = []
        metTrain = []
        times = []
        
        if saveBest:
            shutil.rmtree("model")
            os.mkdir("model")

        self.lr = self.params.learning_rate
        valid = helpers.validSet(self.allPops)

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        optimizer = model.getOptimizer()
        loss = model.getLoss()
        currMet = -1
        bestMet = sys.float_info.max
        if self.metric == "pearson":
            bestMet = sys.float_info.min
        notImprove = 0

        #For n_epochs, train the network iteratively
        for epoch in range(self.n_epochs):
            
            #Decay the learning rate every decay_epoch steps
            if self.lr > 1e-5 and epoch % self.decay_epoch == 0:
                self.lr *= self.lr_decay

            #Optimize the classifier for one step, feeding in relevant info
            X_batch = self.trainInsts
            y_batch = self.trainLabels
            feed_dict = model.createDict(self.keep_prob,X_batch,
                            len(self.trainInsts), None, y_batch, self.lr)
            if self.params.energyOn:
                feed_dict = model.createDict(self.keep_prob,X_batch,
                            len(self.trainInsts), self.energyTrainVec, y_batch, self.lr)
            _,c = sess.run([optimizer, loss], feed_dict=feed_dict)

            #Every calcMetStep, if early stopping or cost outputting is
            #present, do the following
            if epoch % self.calcMetStep == 0:

                #Output the cost and training accuracies
                currMet = -1
                if not production:
                    currMet = self.predict(sess, model, self.testSeqs, goal="testReturn")
                else:
                    currMet = self.predict(sess, model, valid, goal="testReturn")

                if outputCost:
                    metTest.append(currMet)
                    metTrain.append(self.predict(sess, model, \
                                                    self.trainSeqs, goal="trainReturn"))
                    costs.append(c)
                    times.append(epoch)

                #If early stopping is on, find the best metric. If there is no
                #improve for self.nImproveSteps, decay the learning rate
                #significantly. If the training rate drops too low or 
                #The testing accuracy falls deeply below the best metric, stop
                #training
                hasImproved = (self.metric == "pearson" and currMet > bestMet) or \
                       ((self.metric == "md" or self.metric == "rmsd" or \
                           self.metric == "crossEnt") and currMet < bestMet)
                if hasImproved:
                    if self.verbose:
                        print currMet, bestMet
                    bestMet = currMet
                    if saveBest:
                        saver.save(sess, "model/my_model_final.ckpt")
                    notImprove = 0
                else:
                    notImprove += 1
                    if self.verbose:
                        print "Not", notImprove, currMet, bestMet
                if notImprove >= self.nImproveTime/self.calcMetStep:
                    self.lr *= self.largeDecay
                    notImprove = 0
                if self.params.earlyStop and (self.lr <= self.lrThresh or 
                    (not hasImproved and abs(bestMet - currMet) > self.params.metBail)):
                    break

        return times, costs, metTest, metTrain, bestMet

    #Predict the structural ensembles of the predicted sequences passed in.
    #Then, either return this prediction or output it to a file 
    #
    #Params: sess - the session used to make calculations
    #        model - the cnn that has been trained
    #        predSeqs - the sequences the user wants to make predictions of
    #        goal - the users desired. 
    #                    If train or testReturn, return the result
    #                    If best, output results to a file if "other" is < 30
    #                    Otherwise, return the result to a file
    #
    #Return: The predicted structural ensemble if goal=test/trainReturn
    def predict(self, sess, model, predSeqs, goal="test"):
        predVecs = []
        for seq in predSeqs:
            predVec = self.genSeqVec2D(seq, self.numX + self.numExtraX)
            predVecs.append(predVec)
        if len(predVecs) != 0:
            feed_dict = model.createDict(keep_prob=[1.0]*(self.params.numCLayers+1), 
                                         X=predVecs, energy=None, batchSize = len(predSeqs))
            if self.params.energyOn:
                feed_dict = model.createDict(keep_prob=[1.0]*(self.params.numCLayers+1), 
                                         X=predVecs, energy=self.energyTestVec, batchSize = len(predSeqs))
            logits = model.getLogits()
            Z = logits.eval(session=sess, feed_dict=feed_dict)
            result = sess.run(tf.nn.softmax(Z))
            if goal == "best":
                for index, row in enumerate(result):
                    if row[24] * 100 < 30:
                        print predSeqs[index]
                        self.outputPred(row, predSeqs[index], "best")
            elif goal == "testReturn":
                metricSum = 0.0
                for index, row in enumerate(result):
                    metricSum += helpers.calcMetric(row, \
                            self.findTruePop(predSeqs[index]),self.metric)
                metricSum /= float(result.shape[0])
                return metricSum
            elif goal == "trainReturn":
                metricSum = 0.0
                for index, row in enumerate(result):
                    metricSum += helpers.calcMetric(row, \
                            self.findTruePop(predSeqs[index]),self.metric)
                metricSum /= float(result.shape[0])
                return metricSum
            else:
                for index, row in enumerate(result):
                    self.outputPred(row, predSeqs[index], goal)

    #Return the true population of a given sequence
    def findTruePop(self, seq):
        trueRow = []
        for turnComb in self.allTurnCombs:
            if turnComb in self.allPops[seq]:
                trueRow.append(self.allPops[seq][turnComb] / 100.0)
            else:
                trueRow.append(0.0)
        return trueRow

    #Output the predicted and true values for a given structural ensemble a
    #directory indicated by (prefix + "predictions")
    def outputPred(self, pred, seq, prefix):
        predFile = open(prefix + "predictions/Out_prediction_" + seq, "w") 
        if prefix == "best":
            predFile.write('%10s %10s\n' % ('turn comb','pred'))
        else:
            predFile.write('%10s %10s %10s\n' % ('turn comb','pred','true'))
        for index, percent in enumerate(pred):
            turnComb = self.allTurnCombs[index]
            if prefix != "test" and prefix != "train":
                predFile.write('%10s %10.6f\n' % (turnComb, \
                    percent * 100.0))
            else:
                if turnComb in self.allPops[seq]:
                    predFile.write('%10s %10.6f %10.6f\n' % (turnComb, \
                        percent * 100.0,self.allPops[seq][turnComb]))
                else:
                    predFile.write('%10s %10.6f %10.6f\n' % (turnComb, \
                        percent * 100.0,0.0))

        predFile.close()
