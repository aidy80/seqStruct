#Aidan Fike
#June 12, 2019

# Main program. Has options to test the accuracy of a given set of
# hyperparameters of a convolution neural network. It also has the capability
# of predict what sequences will yield well structured cyclic proteins

from cnn import Cnn
from nnUser import nnUser
import parseData
import helpers
import os
import glob
import shutil
import numpy as np
import tensorflow as tf
import hyperparams 
import sys

#Determine the accuracy of a given set of hyperparameters. This is done using
#cross validation on all of the sequences with less than 3 alanines. Also able
#to save the best model found
#
#Params: parameters - The hyperparameters that are used in the following
#                       convolutional net. A params() object
#         testNum - The number that will be saved in the file that is outputted 
#                   in the paramResults directory. An int
#
def createTrainTestPred(parameters, testNum):
    allSeqInfo, allTurnCombs = parseData.getSeqInfo()

    doneSeqs = []

    count = 0
    testSeqs = []
    nn_user = nnUser(parameters)

    #Number of sequences with known structure and less than 3 alanines
    numLessThree = helpers.numValidSet(allSeqInfo) 

    #Print to command line the current parameter information
    helpers.printParamInfo(parameters, testNum) 

    with tf.variable_scope("model" + str(testNum)):
        model = Cnn(parameters)

    bestMetAvg = 0.0
    batchNum = 0
    

    #For each sequence, place it into the current batch. Then, when there are a
    #"batch number" of sequences collected, train a convo network with them as
    #the validation set. Then, measure the accuracy of the current network.
    sess = tf.Session() 
    for seq in allSeqInfo:
        if helpers.numAla(seq) < 3 and seq not in doneSeqs:
            if parameters.verbose:
                print seq
            testSeqs.append(seq)
            count+=1

            doneSeqs.append(seq)
            currSeq = seq
            for i in range(5):
                currSeq = currSeq[5] + currSeq[0:5]
                doneSeqs.append(currSeq)

            if count % parameters.batchSize == 0 or count == numLessThree:
                batchNum += 1
                nn_user.genData(testSeqs, "test")
                times, costs, metTests, metTrains, metTestBest =\
                    nn_user.trainNet(sess,model,outputCost=parameters.outputCost)
                bestMetAvg += metTestBest * len(testSeqs) / helpers.numValidSet(allSeqInfo)
                if parameters.outputCost: #Output files of the training and testing accuracy
                                          #over the training process
                    helpers.outputCostFile(times, costs, testNum, batchNum) 
                    helpers.outputMetTest(times[0:len(metTests)], metTests,
                                            parameters.metric, testNum, batchNum)
                    helpers.outputMetTrain(times[0:len(metTrains)], metTrains,
                                            parameters.metric, testNum, batchNum)

                trainSeqs = helpers.createTrainSeqs(allSeqInfo, testSeqs)
                nn_user.predict(sess, model, testSeqs, "test")
                nn_user.predict(sess, model, trainSeqs, "train")

                first = False
                testSeqs = []
                if not parameters.crossValid:
                    break

    #Output the best metric average and corresponding hyperparameters to a file 
    #in the paramResults directory
    helpers.writeParamInfo("paramResults/","testNum",parameters, bestMetAvg, testNum)
    sess.close()

#Train a cnn with all of the data available. Then, run all possible sequences
#through this trained net and output ones which are well structured - have a
#low unstructured amount
#
#Params: parameters - the hyperparameters that will be used in the algorithm.
#                     a params() object
#
def findWellStruct(parameters):
    allSeqs = helpers.readAllSeqs()
    shutil.rmtree("bestpredictions")
    os.mkdir("bestpredictions")
    
    parameters.earlyStop=False #Necessary. Otherwise, compiler error
    
    nn_user = nnUser(parameters)

    with tf.variable_scope("model", reuse=None):
        model = Cnn(parameters)

    with tf.Session() as sess:
        nn_user.genData([], "test")
        times, costs, metTests, metTrains, metTestBest = nn_user.trainNet(sess, model,\
                                                                            outputCost=False)

        nn_user.predict(sess, model, allSeqs, "best")

#Generate a grid a params() objects and write the results to the
#runScriptParams directory
def gridSearchOutputParams():
    for filename in glob.glob("runScriptParams/*"):
        os.remove(filename)

    allParamSets = hyperparams.searchParams()
    for index, paramSet in enumerate(allParamSets):
        currRun = 1 + helpers.getLastParamNum()
        helpers.writeParamInfo("runScriptParams/","params",paramSet,-1, currRun)
        
#Search all hyperparams in a grid-fashion. The parameters being searched are in
#hyperparams.searchParams(). These are outputted to the
#runScriptParamsDirectory where they can then be evaluated by the
#Sh_parallelMain.sh script
def gridSearch():
    if len(sys.argv) == 3: 
        if str(sys.argv[1]) == "0":
            gridSearchOutputParams()
        else:
            _, parameters = hyperparams.getHyperParams("runScriptParams/params" + \
                                                                sys.argv[1].zfill(3))
            createTrainTestPred(parameters, int(sys.argv[2]) + helpers.getLastTestNum())
     


#Query the paramResults directory to find metric results where certain
#hyperparameters were used
def paramSearch():
    #featNames = ["lr", "pb", "xX", "hn", "fh", "nc", "kp", "ls", "cl", "met", "bs"]
    featNames = ["kp", "fh"]
    featValues = [[0.6,0.6,0.6,0.6], [2, 3]]
    #featNames = ["kp"]
    #featValues = [[0.6,0.6,0.6,0.6]]
    hyperparams.searchParamResults(featNames, featValues)

#Main function. Uncomment relevant lines
def main():
    #parameters = hyperparams.findBestHyper()
    parameters = hyperparams.params()
    createTrainTestPred(parameters, 0)
    #paramSearch() 
    #findWellStruct(parameters)
    #gridSearch()

main()
