import numpy as np

class params():
    learning_rate = 0.001
    n_epochs = 12000
    regBeta = 0.0

    batchSize = 25

    lr_decay=0.97
    decay_epoch=1000
    nImproveTime=2000
    pearBail = 0.013
    lrThresh=1e-5
    largeDecay=0.1
    calcMetStep = 100
    earlyStop=True
    saveBest=False

    numX = 6
    numExtraX = 1

    numHiddenNodes = 0
    numCLayers = 3
    constNumChannels = 64
    numChannels = [constNumChannels]*numCLayers
    filterHeights = [2,3]
    constKeepProb = 0.6
    keep_prob=[constKeepProb]*(numCLayers+1)
    leakSlope = 0.01


def searchParams():
    lr = [0.001]    
    lr_decay = [0.97]
    decay_epoch = [1000]
    pearBail = [0.01]
    numExtraX = [1]
    numHiddenNodes = [0]
    numCLayers = [3]
    numChannels = [64]
    filterHeights = [[2,3], [1,2,3]]
    #keep_prob = [0.6]
    #numCLayers = [3, 5]
    #numChannels = [64, 128]
    #filterHeights = [[2,3], [2,3,4]]
    keep_prob = [0.6, 0.5]
    leakSlope = [0.1, 0.001]

    testingFeatures = [lr, pearBail, numExtraX, numHiddenNodes, 
            filterHeights, numChannels, keep_prob, leakSlope, numCLayers]
    featNames = ["lr", "pb", "xX", "hn", "fh", "nc", "kp", "ls", "cl"]

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

    allParamSets = []
    for row in allParamCombs:
        newParams = params()
        setParams(newParams, featNames, row)
        allParamSets.append(newParams)

    return allParamSets


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
        elif name == "nc":
            params.constNumChannels = featValues[i]
        elif name == "kp":
            params.constKeepProb = featValues[i]
        elif name == "ls":
            params.leakSlope = featValues[i]
        else:
            print name, " is invalid thus far"


def getCurrSubDiv(lenList, totLength, index):
    subDiv = totLength
    for i in range(index):
        subDiv /= lenList[i]
    return subDiv

def makeZerosList(height, width):
    zeros = []
    for i in range(height):
        zeros.append([])
        for j in range(width):
            zeros[i].append(0)
    return zeros
