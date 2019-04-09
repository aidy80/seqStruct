import os
import shutil
import math
import numpy as np

numEnergies = 54

def getTurnCombEn(seq):
    allFiles = os.listdir("allEnergy/" + seq)
    allTurnEnergies = {}
    for currFilename in allFiles:
        energies = getTemplateEnNum()
        with open("allEnergy/" + seq + "/" + currFilename, "r") as currFile:
            for line in currFile:
                words = line.split()
                key = ""
                currEn = 0
                for index, word in enumerate(words):
                    if index == 0:
                        key = word
                    else:
                        currEn += float(word) 
                if len(words) > 1:
                    energies[key] = currEn / float(len(words) - 1)
                elif len(words) == 1:
                    print seq, currFile, "Problem!"

            allTurnEnergies[currFilename.split(".")[0][5:]] = energies

    return allTurnEnergies


def getTemplateEnNum():
    templateEn = {}

    with open("allEnergy/AAADSV/enth_II_0+II_3.dat", "r") as exFile:
        for line in exFile: 
            words = line.split() 
            templateEn[words[0]] = 0
    return templateEn

def getTemplateEnList():
    templateEn = {}

    with open("allEnergy/AAADSV/enth_II_0+II_3.dat", "r") as exFile:
        for line in exFile: 
            words = line.split() 
            templateEn[words[0]] = []
    return templateEn

def getAvgEns(turnCombEn):
    allEns = getTemplateEnList()
    avgEns = {}
    stdEns = {}

    for turnComb in turnCombEn:
        for energy in turnCombEn[turnComb]:
            allEns[energy].append(turnCombEn[turnComb][energy])

    for energy in allEns:
        avg = 0.0
        numVals = 0
        for val in allEns[energy]:
            if val != 0:
                avg += val
                numVals += 1
        avg /= float(numVals)
        avgEns[energy] = avg

    for energy in allEns:
        std = 0.0
        numVals = 0
        for val in allEns[energy]:
            if val != 0:
                std += (val - avgEns[energy])**2
                numVals += 1
        std = math.sqrt(std / float(numVals - 1))
        stdEns[energy] = std

    return avgEns, stdEns

def findMins(turnCombEn):
    mins = []
    for i in range(numEnergies):
        mins.append(1000000)

    for turnComb in turnCombEn:
        for index, energy in enumerate(turnCombEn[turnComb]):
            if turnCombEn[turnComb][energy] < mins[index]:
                mins[index] = turnCombEn[turnComb][energy]
        
    return mins

def getNormEn(avgEns, stdEns, turnCombEn):
    allTurnVals = {}
    mins = findMins(turnCombEn)
    for turnComb in turnCombEn:
        newVals = {}

        for index, energy in enumerate(turnCombEn[turnComb]):
            if turnCombEn[turnComb][energy] == 0:
                #newVals[energy] = np.nan
                newVals[energy] = (mins[index] - avgEns[energy]) / stdEns[energy]
            else: 
                newVals[energy] = (turnCombEn[turnComb][energy] - avgEns[energy]) / stdEns[energy]

        allTurnVals[turnComb] = newVals

    return allTurnVals


def writeOutput(seq, normEns):
    for turnComb in normEns:
        with open("normEn/" + seq + "/enth_" + turnComb + ".dat", "w") as newFile:
            for energy in normEns[turnComb]:
                newFile.write("%12s\t\t%10s\n" % (energy, normEns[turnComb][energy]))  
    
def main():
    thermDirs = os.listdir("allEnergy")

    for seq in thermDirs:
        print seq
        if not os.path.exists("normEn/" + seq):
            os.mkdir("normEn/" + seq)
        turnCombEn = getTurnCombEn(seq) 
        avgEns, stdEns = getAvgEns(turnCombEn)
        normEns = getNormEn(avgEns, stdEns, turnCombEn)
        #for index, key in enumerate(normEns):
        #    print key, normEns[key]
        #    print key, turnCombEn[key]
        #    print "\n"
        writeOutput(seq, normEns)

def rmNums():
    allDirs = os.listdir("OG")
    
    for direc in allDirs:
        shutil.copytree("OG/"+ direc, "OG/"+direc[4:])    
        shutil.rmtree("OG/" + direc)

#rmNums()
main()
