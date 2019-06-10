from cnn import Cnn
from nnUser import nnUser
import parseData
import helpers
import os
import shutil
import numpy as np
import tensorflow as tf

"""
class params():
    learning_rate = 0.001
    n_epochs = 12000
    regBeta = 0.00001
    keep_prob=0.6

    batchSize = 10

    lr_decay=0.99
    decay_epoch=1000
    nImproveTime=300
    lrThresh=1e-5
    largeDecay=0.1
    calcMetStep = 100
    earlyStop=True

    numX = 6

    numHiddenNodes = 0
    numLayers = 1
    numChannels = [64]*numLayers
    filterHeights = [2,3]
"""
class params():
    learning_rate = 0.001
    n_epochs = 12000
    regBeta = 0.00001
    keep_prob=0.6

    batchSize = 10

    lr_decay=0.99
    decay_epoch=1000
    nImproveTime=150
    lrThresh=1e-5
    largeDecay=0.1
    calcMetStep = 50
    earlyStop=True

    numX = 6

    numHiddenNodes = 0
    numLayers = 3
    numChannels = [64]*numLayers
    filterHeights = [2,3]


def createTrainTestPred():
    allSeqInfo, allTurnCombs = parseData.getSeqInfo()

    doneSeqs = []

    count = 0
    testSeqs = []
    parameters = params()
    nn_user = nnUser(parameters)
    numLessThree = helpers.numValidSet(allSeqInfo)

    first = True

    #init = tf.truncated_normal_initializer(stddev=0.1)

    model = Cnn(params)

    sess = tf.Session() 
    for seq in allSeqInfo:
        if helpers.numAla(seq) < 3 and seq not in doneSeqs:
            print seq
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
                    nn_user.trainNet(sess,model,early_stop=params.earlyStop,outputCost=first)
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

def findWellStruct():
    #helpers.genAllSeqs()
    allSeqs = helpers.readAllSeqs()
    shutil.rmtree("bestpredictions")
    os.mkdir("bestpredictions")

    parameters = params()

    test = Cnn(parameters)

    with tf.Session() as sess:
        allSeqVec = []
        for seq in allSeqs:
            allSeqVec.append(test.genSeqVec(seq))
        allSeqVec = np.array(allSeqVec)
        test.genData([], "test")
        times, costs = test.trainNet(sess, allSeqs,[], goal="best")

        costFile = open("costs", "w")
        for time in times:
            costFile.write(str(time) + " " + str(costs[time]) +"\n")
        costFile.close()

def main():
    #findWellStruct()
    createTrainTestPred()

main()
