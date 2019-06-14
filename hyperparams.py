#Aidan Fike
#June 12, 2019

#File to define the hyperparameters used, as well as helper functions to create
#many hyperparameter sets with different combinations of parameters a user is
#curious about testing

import numpy as np
import os

#A class to contain all relevant hyperparameter variables in one place. These
#values are the defaults
class params():
    #The learning rate, number of epochs, and percentage the learning rate
    #decays every decay epoch steps
    learning_rate = 0.001
    n_epochs = 20000
    lr_decay=0.97
    decay_epoch=1000

    #A boolean for whether early stopping should occur, and relevant
    #parameters. This includes, in order, The number of epochs the network has
    #to improve before a largeDecay occurs on the learning rate. Additionally,
    #if the learning rate drops below lrThresh or the validation accuracy falls
    #below pearBail of the best recorded validation accuracy, stop training
    earlyStop=True
    nImproveTime=1500
    largeDecay=0.5
    pearBail = 0.025
    lrThresh=1e-4
    calcMetStep = 100

    #The metric used to evaluate the model. Either pearson, md, or rmsd
    metric = "pearson"

    #The size of the validation set and a boolean for whether cross validation
    #should be used
    batchSize = 27
    crossValid=True

    #Output the best model in the bestModel directory
    saveBest=False

    #The number of amino acids in the instances, and the number of extra amino
    #acids that are added to prevent weirdness of representing a circle as a
    #line
    numX = 6
    numExtraX = 1

    #The variables used the define the network, the number of convolutional
    #layers, the heights of the filters and the number of channels for each
    #filter height. The dropout rate for each layer. The slope of the leaky
    #relu function
    numCLayers = 3
    filterHeights = [2,3]
    constNumChannels = 64
    numChannels = [constNumChannels]*numCLayers
    constKeepProb = 0.6
    keep_prob=[constKeepProb]*(numCLayers+1)
    leakSlope = 0.01
    numHiddenNodes = 0
    regBeta = 0.0

#Create a list of params objects that represent a grid of different hyperparameters
#
#Return: The list of params objects
def searchParams():
    #The hyperparameters that will be used in the grid. If every combination of
    #hyperparameters will be included exactly once
    lr = [0.001]    
    lr_decay = [0.97]
    decay_epoch = [1000]
    pearBail = [0.01]
    numExtraX = [1]
    numHiddenNodes = [0]
    numCLayers = [3,4,5]
    numChannels = [16,32,64]
    filterHeights = [[2,3],[2]]
    keep_prob = [0.6, 0.65]
    leakSlope = [0.01]
    batchSize = [27]

    testingFeatures = [lr, pearBail, numExtraX, numHiddenNodes, batchSize,\
            filterHeights, numChannels, keep_prob, leakSlope, numCLayers]
    featNames = ["lr", "pb", "xX", "hn", "bs", "fh", "nc", "kp", "ls", "cl"]

    #Create the grid of hyperparameters in the form of a 2D list
    paramLenList = []
    totNumCombs = 1
    for feature in testingFeatures:
        paramLenList.append(len(feature))
        totNumCombs *= len(feature)

    allParamCombs = makeZerosList(totNumCombs, len(testingFeatures))
    for featIndex, feature in enumerate(testingFeatures):
        currSubDiv = getCurrSubDiv(paramLenList, totNumCombs, featIndex)
        for i in range(totNumCombs):
            allParamCombs[i][featIndex] = \
                    feature[(i % currSubDiv) / (currSubDiv / len(feature))]

    #Transform the grid of hyperparameters into a set of objects
    allParamSets = []
    allDoneParams = getAllDoneParams()
    for row in allParamCombs:
        newParams = params()
        setParams(newParams, featNames, row)
        notDone = True
        for doneParams in allDoneParams:
            if areEqual(newParams, doneParams):
                notDone = False
        if notDone:
            allParamSets.append(newParams)

    return allParamSets

#Search the paramResults files for results where the given featNames have the
#corresponding featValues. Print the filenames and bestMetric average as the
#result
def searchParamResults(featNames, featValues):
    allFileNamesColl = []
    allBestMets = []
    for index, featName in enumerate(featNames):
        bests, filenames = getParamFiles(featNames[index], featValues[index])
        allFileNamesColl.append(filenames)
        allBestMets.append(bests)

    curatedNames = []
    bestMet = 0.0
    for setNum, fileSet in enumerate(allFileNamesColl):
        for index, name in enumerate(fileSet):
            nameIsValid = True
            for fileSet2 in allFileNamesColl:
                if fileSet != fileSet2:
                    if name not in fileSet2:
                        nameIsValid = False
            if nameIsValid:
                if name not in curatedNames:
                    curatedNames.append(name)
                    bestMet += allBestMets[setNum][index]

    curatedNames.sort()
    print bestMet / float(len(curatedNames)), curatedNames

#Return the name and bestmetric of all parameter files with where the
#passed parameter name has the passed parameter value
def getParamFiles(paramName, paramValue):
    files = os.listdir("paramResults")

    bests = []
    filenames = []

    for filename in files:
        best, params = getHyperParams(filename)
        if paramName == "kp" and params.keep_prob == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "lr" and params.learning_rate == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "xX" and params.numExtraX == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "hn" and params.numHiddenNodes == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "fh" and params.filterHeights == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "nc" and params.numChannels == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "ls" and params.leakSlope == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "cl" and params.numCLayers == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "met" and params.metric == paramValue:
            bests.append(best) 
            filenames.append(filename)
        if paramName == "bs" and params.batchSize == paramValue:
            bests.append(best) 
            filenames.append(filename)


    return bests, filenames

#Return all of the parameter sets who have already been evaluated
def getAllDoneParams():
    files = os.listdir("paramResults")

    allParams = []
    for filename in files:
        best, params = getHyperParams(filename)
        allParams.append(params)
    return allParams

#Check if relevant parameters from two sets of hyperparameters are equal
def areEqual(params1, params2):
    equal = True
    if params1.learning_rate != params2.learning_rate:
        equal = False
    if params1.lr_decay != params2.lr_decay:
        equal = False
    if params1.decay_epoch != params2.decay_epoch:
        equal = False
    if params1.pearBail != params2.pearBail:
        equal = False
    if params1.numExtraX != params2.numExtraX:
        equal = False
    if params1.numHiddenNodes != params2.numHiddenNodes:
        equal = False
    if params1.numCLayers != params2.numCLayers:
        equal = False
    if params1.numChannels != params2.numChannels:
        equal = False
    if params1.filterHeights != params2.filterHeights:
        equal = False
    if params1.keep_prob != params2.keep_prob:
        equal = False
    if params1.leakSlope != params2.leakSlope:
        equal = False
    if params1.batchSize != params2.batchSize:
        equal = False
    if params1.metric != params2.metric:
        equal = False

    return equal

#Take a params object and set the values stored in featNames to the values in
#featValues
#
#Params: params - The params object which will be augmented
#        featNames - codes indicating the names of the features
#        featValues - the values that the features should be set to
def setParams(params, featNames, featValues):
    for i, name in enumerate(featNames):
        if name == "lr":
            params.learning_rate = featValues[i]
        elif name == "pb":
            params.pearBail = featValues[i]
        elif name == "xX":
            params.numExtraX = featValues[i]
        elif name == "hn":
            params.numHiddenNodes = featValues[i]
        elif name == "cl":
            params.numCLayers = featValues[i]
            params.numChannels = [params.constNumChannels]*params.numCLayers
            params.keep_prob= [params.constKeepProb]*(params.numCLayers+1)
        elif name == "fh":
            params.filterHeights = featValues[i]
        elif name == "bs":
            params.batchSize = featValues[i]
        elif name == "nc":
            params.constNumChannels = featValues[i]
        elif name == "kp":
            params.constKeepProb = featValues[i]
        elif name == "ls":
            params.leakSlope = featValues[i]
        elif name == "met":
            params.metric = featValues[i]
        elif name == "bs":
            params.batchSize = featValues[i]
        else:
            print name, " is invalid thus far"

#Helper function to get the current sub division of a list of length totLength.
#The size of the different options are given in lenList 
def getCurrSubDiv(lenList, totLength, index):
    subDiv = totLength
    for i in range(index):
        subDiv /= lenList[i]
    return subDiv

#Create a 2D list of zeros of dimensions height x width 
def makeZerosList(height, width):
    zeros = []
    for i in range(height):
        zeros.append([])
        for j in range(width):
            zeros[i].append(0)
    return zeros

#Find and return the best found set of hyperparameters in paramResults
def findBestHyper():
    files = os.listdir("paramResults")

    bestOfBest = -1
    bestParams = -1
    for filename in files:
        best, params = getHyperParams(filename)
        if best > bestOfBest:
            bestOfBest = best
            bestParams = params

    return bestParams

    
#Read in a parameter test file, return the hyperparameters and best metric
#accuracy
def getHyperParams(filename):
    paramFile = open("paramResults/" + filename, 'r')
    hyper = params()
    bestMet = -1

    for line in paramFile:
        words = line.split()
        if words[0] == "learning_rate":
            hyper.learning_rate = float(words[1])
        elif words[0] == "lr_decay":
            hyper.lr_decay = float(words[1])
        elif words[0] == "decayEpoch":
            hyper.decay_epoch = int(words[1])
        elif words[0] == "nImproveTime":
            hyper.nImproveTime = int(words[1])
        elif words[0] == "lrThresh":
            hyper.lrThresh = float(words[1])
        elif words[0] == "batchSize":
            hyper.batchSize = int(words[1])
        elif words[0] == "largeDecay":
            hyper.largeDecay = float(words[1])
        elif words[0] == "calcMetStep":
            hyper.calcMetStep = int(words[1])
        elif words[0] == "metricBail":
            hyper.pearBail = float(words[1])
        elif words[0] == "numExtraX":
            hyper.numExtraX = int(words[1])
        elif words[0] == "hiddenNodes":
            hyper.numHiddenNodes = int(words[1])
        elif words[0] == "numCLayers":
            hyper.numCLayers = int(words[1])
        elif words[0] == "leakSlope":
            hyper.leakSlope = float(words[1])
        elif words[0] == "metric":
            hyper.metric = words[1]
        elif words[0] == "filterHeights":
            words1 = line.split('[')
            words2 = words1[1].split(',')
            words2[len(words2) - 1] = words2[len(words2) - 1].split(']')[0]
            for index in range(len(words2)):
                words2[index] = int(words2[index])
            hyper.filterHeights = words2
        elif words[0] == "numChannels":
            words1 = line.split('[')
            words2 = words1[1].split(',')
            words2[len(words2) - 1] = words2[len(words2) - 1].split(']')[0]
            for index in range(len(words2)):
                words2[index] = int(words2[index])
            hyper.numChannels = words2
        elif words[0] == "keepProb":
            words1 = line.split('[')
            words2 = words1[1].split(',')
            words2[len(words2) - 1] = words2[len(words2) - 1].split(']')[0]
            for index in range(len(words2)):
                words2[index] = float(words2[index])
            hyper.keep_prob = words2

        elif words[0] == "bestMet":
            bestMet = float(words[1])
        else:
            print filename, words[0], " not a valid option\n"

    paramFile.close()
    return bestMet,hyper
