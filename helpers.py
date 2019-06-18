#Aidan Fike
#June 12, 2019

#Helper functions used in the various other files

import os
import numpy as np
import tensorflow as tf
import math

#Return the number of alanines in a given sequence
def numAla(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

#Return a boolean if the given sequence or a cyclic equivalent exists in the
#testSeqs list
def inSeqs(seq, testSeqs):
    currSeq = seq
    for i in range(len(currSeq)):
        if currSeq in testSeqs:
            return True
        currSeq = currSeq[len(currSeq) - 1] + currSeq[0:len(currSeq) - 1]
    return False

#Create a list of every sequence in allSeqInfo that does not exist in testSeq
def createTrainSeqs(allSeqInfo, testSeqs):
    train = []
    for seq in allSeqInfo:
        if not inSeqs(seq, testSeqs) and numAla(seq) < 3:
            train.append(seq)

    return train

#Output the information in the passed parameters class to the command line
def printParamInfo(parameters, testNum):
    print "\n\n\n"
    print "Running Set number ", testNum
    print "numChannels", parameters.numChannels
    print "leakSlope", parameters.leakSlope
    print "filterHeights", parameters.filterHeights
    print "keepProb", parameters.keep_prob
    print "numCLayers", parameters.numCLayers
    print "metric", parameters.metric
    print "\n"

#Output a file with the costs outputted during the training of a network
def outputCostFile(times, costs):
    costFile = open("costs/cost", "w")
    for index,time in enumerate(times):
        costFile.write(str(time) + " " + str(costs[index]) +"\n")
    costFile.close()

#Output a file with the metric outputted during the training of a network
def outputPTest(times, pTest):
    costFile = open("costs/mTest", "w")
    for index,time in enumerate(times):
        costFile.write(str(time) + " " + str(pTest[index]) +"\n")
    costFile.close()

#Output a file with the metric outputted during the training of a network
def outputPTrain(times, pTrain):
    costFile = open("costs/mTrain", "w")
    for index, time in enumerate(times):
        costFile.write(str(time) + " " + str(pTrain[index]) +"\n")
    costFile.close()

#Output the parameters used in a given run along with the metric result to a
#file in the paramResults directory
def writeParamInfo(path, filename, params, metricResult, testNum):
    paramFile = open(path+filename+str(testNum).zfill(3), "w")
    
    paramFile.write("learning_rate " + str(params.learning_rate) + "\n")
    paramFile.write("lr_decay " + str(params.lr_decay) + "\n")
    paramFile.write("decayEpoch " + str(params.decay_epoch) + "\n")
    paramFile.write("nImproveTime " + str(params.nImproveTime) + "\n")
    paramFile.write("lrThresh " + str(params.lrThresh) + "\n")
    paramFile.write("largeDecay " + str(params.largeDecay) + "\n")
    paramFile.write("calcMetStep " + str(params.calcMetStep) + "\n")
    paramFile.write("metricBail " + str(params.pearBail) + "\n")
    paramFile.write("numExtraX " + str(params.numExtraX) + "\n")
    paramFile.write("hiddenNodes " + str(params.numHiddenNodes) + "\n")
    paramFile.write("numCLayers " + str(params.numCLayers) + "\n")
    paramFile.write("filterHeights " + str(params.filterHeights) + "\n")
    paramFile.write("numChannels " + str(params.numChannels) + "\n")
    paramFile.write("keepProb " + str(params.keep_prob) + "\n")
    paramFile.write("leakSlope " + str(params.leakSlope) + "\n")
    paramFile.write("metric " + str(params.metric) + "\n")
    paramFile.write("batchSize " + str(params.batchSize) + "\n")
    
    if metricResult != -1:
        paramFile.write("bestMet " + str(metricResult) + "\n")

    paramFile.close()

#Return the number of sequences with known structures that have less than 3
#alanines. Cyclic equivalents not double counted
def numValidSet(allSeqInfo):
    doneSeqs = []
    
    count = 0

    for seq in allSeqInfo:
        if numAla(seq) < 3 and seq not in doneSeqs:
            count += 1
            doneSeqs.append(seq)
            currSeq = seq
            for i in range(5):
                currSeq = currSeq[5] + currSeq[0:5]
                doneSeqs.append(currSeq)

    return count

#Calculate a given metric between two sequences of values, one intended to be
#predicted and the other intended to be true. Options are pearson correlation 
#coefficient, rmsd, or mean displacement (md)
def calcMetric(row1, row2, metric):
    preds = []
    true = []
    #Include everything but the values for the "others" catagory
    for index, val in enumerate(row1):
        if index != 24:
            preds.append(row1[index] * 100.0)
            true.append(row2[index] * 100.0)
                
    #Calculate and return the pearson correlation coefficient
    if metric == "pearson":
        meanY = float(sum(true)) / float(len(true))
        meanX = float(sum(preds)) / float(len(preds))
        varX = 0.0
        varY = 0.0
        covar = 0.0

        for index, pred in enumerate(preds):
            covar += (pred - meanX) * (true[index] - meanY)
            varX += (pred - meanX)**2 
            varY += (true[index] - meanY)**2 
        varX /= float(len(preds) - 1)
        varY /= float(len(preds) - 1)

        pCoef = covar / ((float(len(preds)) - 1.0) * math.sqrt(varX * varY)) 
        return pCoef

    #Calculate and return the rmsd
    if metric == "rmsd":
        rmsd = 0.0
        
        for index, pred in enumerate(preds):
            rmsd += (true[index] - pred)**2
        rmsd = math.sqrt(rmsd / float(len(preds)))
        return rmsd

    #Calculate and return the mean displacement
    if metric == "md":
        straightMd = 0.0

        for index, pred in enumerate(preds):
            straightMd += abs(true[index] - pred)

        md = straightMd / float(len(preds))
        return md

#Go through the paramResults directory and find the last testNum that was done
def getLastTestNum():
    lastTestNum = 0
    for name in os.listdir("paramResults"):
        if int(name[7:]) > lastTestNum:
            lastTestNum = int(name[7:])

    return lastTestNum
    
#Go through the runScripts/params directory and find the largest paramsNum 
def getLastParamNum():
    lastTestNum = 0
    for name in os.listdir("runScriptParams"):
        if int(name[7:]) > lastTestNum:
            lastTestNum = int(name[7:])

    return lastTestNum
 
    
#Generate all possible sequences of hexapeptides and 
def genAllSeqs():
    aminos = parseData.getAminos()
    numX = 6
    allSeqs = []

    #Add every combination of sequences
    for m in range(len(aminos)):
        allSeqs.append(aminos[m])

    for i in range(numX - 1):
        newAllSeqs = []
        for partSeq in allSeqs:
            for j in range(len(aminos)):
                newAllSeqs.append(partSeq + aminos[j])     
        allSeqs = newAllSeqs

    deletedSeqs = []
    
    #Remove cyclic equivalents
    for seq in allSeqs:
        if len(seq) < numX:
            allSeqs.remove(seq)
        else:
            currSeq = seq
            for i in range(numX - 1):
                currSeq = currSeq[numX-1] + currSeq[0:numX - 1]
                if currSeq != seq and currSeq not in deletedSeqs:
                    deletedSeqs.append(currSeq)
                    allSeqs.remove(currSeq)

    #Output all of the possible sequences into the allSeqs file
    allSeqFile = open("allSeqs", "w")
    for seq in allSeqs:
        allSeqFile.write(seq + "\n")
    allSeqFile.close()

#Read and return a list of the sequences in the allSeqs file
def readAllSeqs():
    allSeqs = []

    allSeqFile = open("allSeqs", "r")
    for line in allSeqFile:
        words = line.split()
        allSeqs.append(words[0])

    return allSeqs
