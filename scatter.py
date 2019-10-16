#Aidan Fike
#June 12, 2019

#Plot data predicted from the most recent neural network training/testing session.
#This can be done with either testing or training data by running
#
#python scatter.py test
#or
#python scatter.py train

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import math
import os
import matplotlib.colors as color
import itertools
import sys

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 28})

allFiles = os.listdir(sys.argv[1] + "predictions")

#Return the number of alanine in a passed sequence
def numA(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

pCoefSum = 0.0
rmsdSum = 0.0
numNames = 0
mdSum = 0.0

#allFiles = ["Out_prediction_NSAAAG", "Out_prediction_RAAAAS"]
#allFiles = ["Out_prediction_FRSANR"]

minName =''
minP = 1.0
maxName=''
maxP = -1.0

#For each predicted structural ensemble with less than 3 alanine, calculate its 
#pearson correlation coefficient, rmsd, and md (mean displacement). 
#Additionally, plot the predicted in sequence information.
for name in allFiles:
    print name
    preds = []
    predY = []
    true = []

    normSeqName = name[15:21]

    predFile = open(sys.argv[1] + "predictions/" + name, "r")

    for line in predFile:
        words = line.split()
        if words[0] != "turn" and words[0] != 'mse:' and words[0] != 'other':
            preds.append(float(words[1]))
            predY.append(float(words[1]))
            true.append(float(words[2]))
                
    meanY = float(sum(true)) / float(len(true))
    meanX = float(sum(preds)) / float(len(preds))
    varX = 0.0
    varY = 0.0
    covar = 0.0

    rmsd = 0.0
    straightMd = 0.0

    for index, pred in enumerate(preds):
        covar += (pred - meanX) * (true[index] - meanY)
        varX += (pred - meanX)**2 
        varY += (true[index] - meanY)**2 
        rmsd += (true[index] - pred)**2
        straightMd += abs(true[index] - pred)

    rmsd = math.sqrt(rmsd / float(len(preds)))
    rmsdSum += rmsd
    md = straightMd / float(len(preds))
    mdSum += md
    print("rmsd: " + name + " " + str(rmsd))
    print("md: " + name + " " + str(md))

    varX /= float(len(preds) - 1)
    varY /= float(len(preds) - 1)

    pCoef = covar / ((float(len(preds)) - 1.0) * math.sqrt(varX * varY)) 

    if numA(normSeqName) < 3:
        pCoefSum+=pCoef
        numNames+=1
        print("pearson: " + name + " " + str(pCoef))
    
    if normSeqName == "NADRNV":
        font = {'family' : 'normal', 
                'size'   : 28}
        matplotlib.rc('font', **font)

        plt.title("Predictions of %s" % normSeqName)
        plt.xlabel("Predicted TurnComb \n Likelihoods (%)")
        plt.ylabel("BeMeta TurnComb \n Likelihoods (%)")
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
        yPos=max(true) - 0.4
        label = "p = %5.3f" % pCoef
        #plt.annotate(label, xy=(xPos, yPos)) 
        plt.tight_layout()
        preds.append(max(preds)+.6)
        predY.append(max(predY)+.6)
        plt.plot(np.array(preds), np.array(predY), label = 'Line of Perfect Prediction')
        plt.show()


    print
    predFile.close()

#print "max: ", maxP, maxName
#print "min: ", minP, minName
print("rmsdAvg:", rmsdSum / float(numNames))
print("pCoefAvg: ", pCoefSum / float(numNames))
print("mdAvg:", mdSum / float(numNames))
