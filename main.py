from cnn import Cnn
from nnUser import nnUser
import parseData
import helpers
import os
import shutil
import numpy as np
import tensorflow as tf
from hyperparams import params

class defaultParams():
    learning_rate = 0.001
    n_epochs = 12000
    regBeta=0.0

    batchSize = 10

    lr_decay=0.99
    decay_epoch=1000
    nImproveTime=2000
    pearBail = 0.013
    lrThresh=1e-5
    largeDecay=0.1
    calcMetStep = 100
    earlyStop=True
    saveBest=False

    numX = 6

    numHiddenNodes = 0
    numCLayers = 3
    numChannels = [64]*numCLayers
    filterHeights = [2,3]
    keep_prob=[0.6]*(numCLayers+1)
    leakSlope = 0.01

def createTrainTestPred(parameters):
    allSeqInfo, allTurnCombs = parseData.getSeqInfo()

    doneSeqs = []

    count = 0
    testSeqs = []
    nn_user = nnUser(parameters)
    numLessThree = helpers.numValidSet(allSeqInfo)

    first = True

    with tf.variable_scope("model", reuse=None):
        model = Cnn(parameters)

    sess = tf.Session() 
    for seq in allSeqInfo:
        if helpers.numAla(seq) < 3 and seq not in doneSeqs:
            print(seq)
            testSeqs.append(seq)
            count+=1

            doneSeqs.append(seq)
            currSeq = seq
            for i in range(5):
                currSeq = currSeq[5] + currSeq[0:5]
                doneSeqs.append(currSeq)

            if count % parameters.batchSize == 0 or count == numLessThree:
                nn_user.genData(testSeqs, "test")
                times, costs, pTests, pTrains =\
                    nn_user.trainNet(sess,model,outputCost=first)
                if first:
                    helpers.outputCostFile(times, costs) 
                    helpers.outputPTest(times[0:len(pTests)], pTests)
                    helpers.outputPTrain(times[0:len(pTrains)], pTrains)

                trainSeqs = helpers.createTrainSeqs(allSeqInfo, testSeqs)
                nn_user.predict(sess, model, testSeqs, "test")
                nn_user.predict(sess, model, trainSeqs, "train")

                first = False
                testSeqs = []

    sess.close()

def findWellStruct(parameters):
    #helpers.genAllSeqs()
    allSeqs = helpers.readAllSeqs()
    shutil.rmtree("bestpredictions")
    os.mkdir("bestpredictions")
    
    parameters.earlyStop=False
    
    nn_user = nnUser(parameters)

    with tf.variable_scope("model", reuse=None):
        model = Cnn(parameters)

    with tf.Session() as sess:
        print "one"
        nn_user.genData([], "test")
        print "two"
        times, costs, pTests, pTrains = nn_user.trainNet(sess, model,\
                outputCost=False)
        print "three"

        nn_user.predict(sess, model, allSeqs, "best")
        
def main():
    parameters = defaultParams()
    findWellStruct(parameters)
    #createTrainTestPred(parameters)

main()
