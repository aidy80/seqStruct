#Aidan Fike
#June 12, 2019

#File to define the hyperparameters used, as well as helper functions to create
#many hyperparameter sets with different combinations of parameters a user is
#curious about testing

import numpy as np

#A class to contain all relevant hyperparameter variables in one place. These
#values are the defaults
class params():
    #The learning rate, number of epochs, and percentage the learning rate
    #decays every decay epoch steps
    learning_rate = 0.001
    n_epochs = 15000
    lr_decay=0.97
    decay_epoch=1000



    #A boolean for whether early stopping should occur, and relevant
    #parameters. This includes, in order, The number of epochs the network has
    #to improve before a largeDecay occurs on the learning rate. Additionally,
    #if the learning rate drops below lrThresh or the validation accuracy falls
    #below pearBail of the best recorded validation accuracy, stop training
    earlyStop=True
    nImproveTime=1500
    largeDecay=0.1
    pearBail = 0.013
    lrThresh=1e-5
    calcMetStep = 100

    #The size of the validation set and a boolean for whether cross validation
    #should be used
    batchSize = 25
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
    numCLayers = [3,4]
    numChannels = [16,64,128]
    filterHeights = [[2,3],[2,3,4]]
    keep_prob = [0.6, 0.7]
    leakSlope = [0.01]

    testingFeatures = [lr, pearBail, numExtraX, numHiddenNodes, 
            filterHeights, numChannels, keep_prob, leakSlope, numCLayers]
    featNames = ["lr", "pb", "xX", "hn", "fh", "nc", "kp", "ls", "cl"]

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
    for row in allParamCombs:
        newParams = params()
        setParams(newParams, featNames, row)
        allParamSets.append(newParams)

    return allParamSets

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
        elif name == "nc":
            params.constNumChannels = featValues[i]
        elif name == "kp":
            params.constKeepProb = featValues[i]
        elif name == "ls":
            params.leakSlope = featValues[i]
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
