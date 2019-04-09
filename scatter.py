import matplotlib.pyplot as plt
import numpy as np
import math
import math
import os
import matplotlib.colors as color
import itertools

allFiles = os.listdir("predictions")

domain = []
xEqualY = []

def numA(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

for i in range(1000):
    domain.append(float(i) / 100.0)
    xEqualY.append(float(i) / 100.0)

pCoefSum = 0.0
numNames = 0

#allFiles = ["Out_prediction_NSAAAG", "Out_prediction_RAAAAS"]
#allFiles = ["Out_prediction_FRSANR"]

minName =''
minP = 1.0
maxName=''
maxP = -1.0

for name in allFiles:
    print name
    normSeqName = name[15:21]

    predFile = open("predictions/" + name, "r")

    preds = []
    true = []
    predY = []
    predSum = 0.0
    for line in predFile:
        words = line.split()
        if words[0] != "turn" and words[0] != 'mse:' and words[0] != 'other':
        #if words[0] != "turn" and words[0] != 'mse:':
            predSum += float(words[1])
            if words[0] != "other":
                preds.append(float(words[1]))
                predY.append(float(words[1]))
                true.append(float(words[2]))
                
    meanY = float(sum(true)) / float(len(true))
    meanX = float(sum(preds)) / float(len(preds))
    varX = 0.0
    varY = 0.0
    covar = 0.0

    for index, pred in enumerate(preds):
        covar += (pred - meanX) * (true[index] - meanY)
        varX += (pred - meanX)**2 
        varY += (true[index] - meanY)**2 

    varX /= len(preds) - 1
    varY /= len(preds) - 1

    pCoef = covar / ((len(preds) - 1) * math.sqrt(varX * varY)) 
    """
    if (pCoef > maxP):
        maxP = pCoef
        maxName = name
    if (pCoef < minP):
        minP = pCoef
        minName = name
    """

    if numA(normSeqName) < 3:
        pCoefSum+=pCoef
        numNames+=1
        print(name, pCoef)

    
    """
    if normSeqName == "GNAAAA":
        plt.title("Predictions of %s" % normSeqName)
        plt.xlabel("Predicted Turn Combination Likelihoods (%)")
        plt.ylabel("BeMeta Turn Combination Likelihoods (%)")
        plt.xlim(0,max(preds)+.5)
        plt.ylim(0,max(true)+.5)
        colorVals = []
        dists = []
        for index, pred in enumerate(preds):
            dists.append(abs(pred-true[index]))

        for dist in dists:
            colorVals.append((dist/max(dists),0,0))
        colors = itertools.cycle(colorVals)
        for index, pred in enumerate(preds):
            plt.scatter(pred, true[index], color=next(colors))
        xPos=0.5
        yPos=max(true) - 1
        label = "rho = %5.3f" % pCoef
        plt.annotate(label, xy=(xPos, yPos)) 
        preds.append(max(preds)+.6)
        predY.append(max(predY)+.6)
        plt.plot(np.array(preds), np.array(predY), label = 'Line of Perfect Prediction')
        plt.show()
    """

    predFile.close()

#print "max: ", maxP, maxName
#print "min: ", minP, minName
print("pCoefAvg: ", pCoefSum / float(numNames))
