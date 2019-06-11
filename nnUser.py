import parseData
import helpers
import os
import numpy as np
import tensorflow as tf

class nnUser():
    def __init__(self, params): 
        self.allPops, self.allTurnCombs = parseData.getSeqInfo()
        self.allTurnCombs = self.allTurnCombs.keys()
        self.aminos = parseData.getAminos()
        self.numX = params.numX

        self.trainInsts = []
        self.trainLabels = []
        self.testInsts = []
        self.testLabels = []
        self.testSeqs = []
        self.trainSeqs = []

        self.params = params
        
        self.calcMetStep = params.calcMetStep
        self.lr_decay = params.lr_decay
        self.lr = params.learning_rate
        self.lrThresh = params.lrThresh
        self.largeDecay = params.largeDecay
        self.keep_prob = params.keep_prob
        self.n_epochs = params.n_epochs
        self.decay_epoch = params.decay_epoch
        self.nImproveTime = params.nImproveTime

    def resetTrainTest(self):
        self.testSeqs = []
        self.trainSeqs = []

        self.trainInsts = []
        self.trainLabels = []
        self.testInsts = []
        self.testLabels = []

    def genData(self, testSeqs, dataType):
        self.resetTrainTest()
        self.testSeqs = testSeqs
        self.trainSeqs = helpers.createTrainSeqs(self.allPops, testSeqs)

        turnCount = 0

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
                    seqVec = self.genSeqVec2D(seq, self.numX + 1)
                    if not helpers.inSeqs(seq, testSeqs) or dataType == "train":
                        self.trainInsts.append(seqVec)
                    elif seq in testSeqs:
                        self.testInsts.append(seqVec)

            turnCount+=1

        self.trainLabels = np.array(self.trainLabels).transpose()
        self.testLabels = np.array(self.testLabels).transpose()
        self.trainInsts = np.array(self.trainInsts)
        self.testInsts = np.array(self.testInsts)
        
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

    def genSeqVec(self, seq, vecLength):
        inv_aminos_map = {v: k for k, v in self.aminos.iteritems()} 
        seq_vec = np.zeros(len(self.aminos) * vecLength)
        for index, amino in enumerate(seq):
            if (index < self.numX):
                seq_vec[inv_aminos_map[amino] + index*len(self.aminos)] = 1
        for index, amino in enumerate(seq):
            if (index < vecLength - self.numX):
                seq_vec[inv_aminos_map[amino] + (index + self.numX)*len(self.aminos)] = 1

        return seq_vec 

    def trainNet(self, sess, model, outputCost):
        costs = []
        metTest = []
        metTrain = []
        times = []
        
        self.lr = self.params.learning_rate

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        optimizer = model.getOptimizer()
        loss = model.getLoss()
        bestMet = -1
        currMet = -1
        notImprove = 0

        for epoch in range(self.n_epochs):
            if self.lr > 1e-5 and epoch % self.decay_epoch == 0:
                self.lr *= self.lr_decay
            X_batch = self.trainInsts
            y_batch = self.trainLabels
            feed_dict = model.createDict(self.keep_prob, X_batch,
                    len(self.trainInsts), y_batch,
                                self.lr)
            _,c = sess.run([optimizer, loss], feed_dict=feed_dict)
            if epoch % self.calcMetStep == 0 and (self.params.earlyStop or outputCost):
                currMet = self.predict(sess, model, self.testSeqs, goal="testReturn")
                print "currMet: ", currMet
                if outputCost:
                    metTest.append(currMet)
                    metTrain.append(self.predict(sess, model, \
                                                    self.trainSeqs, goal="trainReturn"))
                if(self.params.earlyStop):
                    if currMet > bestMet:
                        print currMet, bestMet
                        bestMet = currMet
                        if self.params.saveBest:
                            saver.save(sess, "model/my_model_final.ckpt")
                        notImprove = 0
                    else:
                        notImprove += 1
                        print "Not", notImprove, currMet, bestMet
                    if notImprove >= self.nImproveTime/self.calcMetStep:
                        self.lr *= self.largeDecay
                        notImprove = 0
                    if self.lr <= self.lrThresh or bestMet - currMet > self.params.pearBail:
                        break

            costs.append(c)
            times.append(epoch)
        
        return times, costs, metTest, metTrain

    def predict(self, sess, model, predSeqs, goal="test"):
        predVecs = []
        for seq in predSeqs:
            predVec = self.genSeqVec2D(seq, self.numX + 1)
            predVecs.append(predVec)
        if len(predVecs) != 0:
            feed_dict = model.createDict(keep_prob=[1.0]*(self.params.numCLayers+1), X=predVecs,\
                                                    batchSize = len(predSeqs))
            logits = model.getLogits()
            Z = logits.eval(session=sess, feed_dict=feed_dict)
            result = sess.run(tf.nn.softmax(Z))
            if goal == "best":
                for index, row in enumerate(result):
                    if row[24] * 100 < 25:
                        print predSeqs[index]
                        self.outputPred(row, predSeqs[index], "best")
            elif goal == "testReturn":
                metricSum = 0.0
                for index, row in enumerate(result):
                    metricSum += helpers.calcMetric(row, \
                            self.findTruePop(predSeqs[index]),"pearson")
                metricSum /= float(result.shape[0])
                return metricSum
            elif goal == "trainReturn":
                metricSum = 0.0
                for index, row in enumerate(result):
                    metricSum += helpers.calcMetric(row, \
                            self.findTruePop(predSeqs[index]),"pearson")
                metricSum /= float(result.shape[0])
                return metricSum
            else:
                for index, row in enumerate(result):
                    self.outputPred(row, predSeqs[index], goal)

    def findTruePop(self, seq):
        trueRow = []
        for turnComb in self.allTurnCombs:
            if turnComb in self.allPops[seq]:
                trueRow.append(self.allPops[seq][turnComb])
            else:
                trueRow.append(0.0)
        return trueRow

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
