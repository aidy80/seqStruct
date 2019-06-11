import os
import numpy as np
import tensorflow as tf
import math

#cnnHelpers

def numAla(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

def inSeqs(seq, testSeqs):
    currSeq = seq
    for i in range(len(currSeq)):
        if currSeq in testSeqs:
            return True
        currSeq = currSeq[len(currSeq) - 1] + currSeq[0:len(currSeq) - 1]
    return False

def createTrainSeqs(allSeqInfo, testSeqs):
    train = []
    for seq in allSeqInfo:
        if not inSeqs(seq, testSeqs) and numAla(seq) < 3:
            train.append(seq)

    return train

def outputCostFile(times, costs):
    costFile = open("costs/cost", "w")
    for time in times:
        costFile.write(str(time) + " " + str(costs[time]) +"\n")
    costFile.close()

def outputPTest(times, pTest):
    costFile = open("costs/mTest", "w")
    for time in times:
        costFile.write(str(time) + " " + str(pTest[time]) +"\n")
    costFile.close()

def outputPTrain(times, pTrain):
    costFile = open("costs/mTrain", "w")
    for time in times:
        costFile.write(str(time) + " " + str(pTrain[time]) +"\n")
    costFile.close()

def outputParamResults(params, metricResult, testNum):
    paramFile = open("paramResults/testNum"+str(testNum), "w")
    
    paramFile.write("learning_rate " + str(params.learning_rate) + "\n")
    paramFile.write("metricBail " + str(params.pearBail) + "\n")
    paramFile.write("numExtraX " + str(params.numExtraX) + "\n")
    paramFile.write("hiddenNodes " + str(params.numHiddenNodes) + "\n")
    paramFile.write("convLayers " + str(params.numCLayers) + "\n")
    paramFile.write("filterHeights " + str(params.filterHeights) + "\n")
    paramFile.write("numChannels " + str(params.numChannels) + "\n")
    paramFile.write("keepProb " + str(params.keep_prob) + "\n")
    paramFile.write("leakSlope " + str(params.leakSlope) + "\n")
    
    paramFile.write("bestMet " + str(metricResult) + "\n")

    paramFile.close()

    return testNum

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

def calcMetric(row1, row2, metric):
    preds = []
    true = []
    for index, val in enumerate(row1):
        if index != 24:
            preds.append(row1[index] * 100.0)
            true.append(row2[index] * 100.0)
                
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

    if metric == "rmsd":
        rmsd = 0.0
        
        for index, pred in enumerate(preds):
            rmsd += (true[index] - pred)**2
        rmsd = math.sqrt(rmsd / float(len(preds)))
        return rmsd

    if metric == "md":
        straightMd = 0.0

        for index, pred in enumerate(preds):
            straightMd += abs(true[index] - pred)

        md = straightMd / float(len(preds))
        return md
    
def genAllSeqs():
    aminos = parseData.getAminos()
    numX = 6
    allSeqs = []
    for m in range(len(aminos)):
        allSeqs.append(aminos[m])

    for i in range(numX - 1):
        newAllSeqs = []
        for partSeq in allSeqs:
            for j in range(len(aminos)):
                newAllSeqs.append(partSeq + aminos[j])     
        allSeqs = newAllSeqs

    deletedSeqs = []
    
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

    allSeqFile = open("allSeqs", "w")
    for seq in allSeqs:
        allSeqFile.write(seq + "\n")
    allSeqFile.close()

def readAllSeqs():
    allSeqs = []

    allSeqFile = open("allSeqs", "r")
    for line in allSeqFile:
        words = line.split()
        allSeqs.append(words[0])

    return allSeqs

def outputPred(pred, seq, prefix):
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
