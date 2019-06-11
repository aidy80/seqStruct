from cnn import Cnn
from nnUser import nnUser
import parseData
import helpers
import os
import shutil
import numpy as np
import tensorflow as tf
import hyperparams 

def createTrainTestPred(parameters, testNum):
    allSeqInfo, allTurnCombs = parseData.getSeqInfo()

    doneSeqs = []

    count = 0
    testSeqs = []
    nn_user = nnUser(parameters)
    numLessThree = helpers.numValidSet(allSeqInfo)

    first = True

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        model = Cnn(parameters)

    bestMetAvg = 0.0

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
                times, costs, metTests, metTrains, metTestBest =\
                    nn_user.trainNet(sess,model,outputCost=first)
                bestMetAvg += metTestBest / len(testSeqs)
                if first:
                    helpers.outputCostFile(times, costs) 
                    helpers.outputPTest(times[0:len(metTests)], metTests)
                    helpers.outputPTrain(times[0:len(metTrains)], metTrains)

                trainSeqs = helpers.createTrainSeqs(allSeqInfo, testSeqs)
                nn_user.predict(sess, model, testSeqs, "test")
                nn_user.predict(sess, model, trainSeqs, "train")

                first = False
                testSeqs = []

    helpers.outputParamResults(parameters, bestMetAvg, testNum)
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
        nn_user.genData([], "test")
        times, costs, metTests, metTrains, metTestBest = nn_user.trainNet(sess, model,\
                                                                            outputCost=False)

        nn_user.predict(sess, model, allSeqs, "best")
        
def main():
    parameters = hyperparams.params()
    allParamSets = hyperparams.searchParams()
    #allParamSets = [parameters]
    for index, paramSet in enumerate(allParamSets):
        createTrainTestPred(paramSet, index + 1)

    #findWellStruct(parameters)

main()
