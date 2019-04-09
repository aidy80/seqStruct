import os

def getAminos():
    return  {0: 'A', 1: 'G', 2: 'V', 3: 'S', 4: 'F', 5: 'N', 6: 'R', 7: 'D'}

def getEnergyInfo():
    therm = os.listdir("energy/normEn/")
    thermDirs = []
    energyInfo = {}

    for thermDir in therm:
        thermDirs.append(thermDir)
        energyInfo[thermDir] = {}

    for thermDir in thermDirs:
        seqName = thermDir
        allThermFiles = os.listdir("energy/normEn/" + thermDir)
        for thermFile in allThermFiles:
            if os.path.getsize("energy/normEn/" + thermDir + "/" +
                    thermFile) >= 1000:
                turnComb = thermFile[5:].split("+")
                turnComb = (turnComb[0], turnComb[1].split(".")[0])
                turnComb = turnComb[0] + turnComb[1]
                energyInfo[seqName][turnComb] = {}
                with open("energy/normEn/" + thermDir + "/" + thermFile, "r") as currFile:
                    for line in currFile:
                        ener = line.split()
                        enerSum = 0.0
                        for index, amount in enumerate(ener):
                            if index!=0:
                                enerSum += float(amount)
                        energyInfo[seqName][turnComb][ener[0]] = enerSum / \
                                                                    float(len(ener) - 1)

    return energyInfo

def getSolEnergyInfo():
    therm = os.listdir("energy/allEnergy/")
    thermDirs = []
    energyInfo = {}

    for thermDir in therm:
        thermDirs.append(thermDir)
        energyInfo[thermDir] = {}

    for thermDir in thermDirs:
        seqName = thermDir
        allThermFiles = os.listdir("energy/allEnergy/" + thermDir)
        for thermFile in allThermFiles:
            turnComb = thermFile[5:].split("+")
            turnComb = (turnComb[0], turnComb[1].split(".")[0])
            turnComb = turnComb[0] + turnComb[1]
            energyInfo[seqName][turnComb] = {}

            if os.path.getsize("energy/allEnergy/" + thermDir + "/" +
                    thermFile) >= 200:
                with open("energy/allEnergy/" + thermDir + "/" + thermFile, "r") as currFile:
                    numToEn = {}
                    numEnLines = 0
                    for line in currFile:
                        ener = line.split()
                        #if ener[0] == "#" and ener[1] != "ClusterID" and numEnLines == 0:
                        if ener[0] == "#" and ener[1] != "ClusterID" and numEnLines == 0:
                            counter = 0
                            for en in ener:
                                if en != "#":
                                    numToEn[counter] = en
                                    energyInfo[seqName][turnComb][en] = 0
                                    counter+=1
                                
                        elif ener[0][0] != "#":
                            numEnLines+=1
                            for index, amount in enumerate(ener):
                                energyInfo[seqName][turnComb][numToEn[index]]\
                                                            += float(amount)
                    for en in energyInfo[seqName][turnComb]:
                        energyInfo[seqName][turnComb][en] /= numEnLines

    return energyInfo

def getSeqInfo3A():
    allFiles = os.listdir("data")

    seqs = []

    for name in allFiles:
        normSeqName = name[15:21]
        seqs.append(normSeqName)

    allSeqInfo = {}
    allTurnCombs = {}
    for index, name in enumerate(allFiles):
        seqPopSum = 0.0
        popFile = open("data/" + name, "r")
        seqInfo = {}
        for line in popFile:
            words = line.split()
            turnComb = words[0] + words[1]

            if words[0] != "turn":
                if turnComb in allTurnCombs:
                    allTurnCombs[turnComb] += 1
                else:
                    allTurnCombs[turnComb] = 0 

                seqInfo[turnComb] = float(words[3])
                seqPopSum += float(words[3])
        seqInfo["other"] = 100.0 - seqPopSum
        allSeqInfo[seqs[index]] = seqInfo
        popFile.close()
    allTurnCombs["other"] = len(seqs)

    return allSeqInfo, allTurnCombs


def getTurnCombPops(allSeqInfo, turnComb):
    turnPops = {}
    for seq in allSeqInfo:
        turns = allSeqInfo[seq]
        if turnComb in turns:
            turnPops[seq] = turns[turnComb]
        else:
            turnPops[seq] = 0

    return turnPops

#allSeq, allTurn = getSeqInfo3A()
#energyInfo = getEnergyInfo()
#energyInfo = getSolEnergyInfo()
