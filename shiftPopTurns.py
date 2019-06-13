#Aidan Fike
#June 12, 2019

#Program to create cyclic equivalents for all of the structural ensembles found
#through Be-Meta

import os

def numA(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

def shiftLeft(seq, numShifts, aug):
    if numShifts == 0:
        seqInfo = readFile(seq, aug) 
        writeSeq(seq, seqInfo)

    elif numShifts == 3:  
        seqInfo = readFile(seq, aug) 
        newSeqInfo = {}

        for turnComb in seqInfo:
            turns = turnComb.split()
            newTurnComb = turns[1][:len(turns[1]) - 1] + \
                          turns[0][len(turns[0]) - 1]  + " " + \
                          turns[0][:len(turns[0]) - 1] + \
                          turns[1][len(turns[1]) - 1]
            newSeqInfo[newTurnComb] = seqInfo[turnComb]

        newSeq = seq[3:6] + seq[0:3]
        writeSeq(newSeq, newSeqInfo)

    elif numShifts == 1:
        seqInfo = readFile(seq, aug) 
        newSeqInfo = {}

        for turnComb in seqInfo:
            turns = turnComb.split()
            turnType1 = turns[0][:len(turns[0]) - 2]
            turnNum1 = int(turns[0][len(turns[0]) - 1])
            turnType2 = turns[1][:len(turns[1]) - 2]
            turnNum2 = int(turns[1][len(turns[1]) - 1])
            
            turnNum2 = turnNum2 - 1
            turnNum1 = turnNum1 - 1
            newTurnComb = turnType1 + "_" + str(turnNum1) + " " + turnType2 + "_" + str(turnNum2)

            if (turnNum1 == -1):
                turnNum2 = 5
                turnNum1 = 2
                newTurnComb = turnType2 + "_" + str(turnNum1) + " " + turnType1\
                + "_" + str(turnNum2) #NOT A WEIRD TYP0 weird switch happens

            newSeqInfo[newTurnComb] = seqInfo[turnComb]

        newSeq = seq[1:6] + seq[0]
        writeSeq(newSeq, newSeqInfo)

    else:
        print "Not yet implemented\n"

def shiftRight(seq, numShifts, aug):
    if numShifts == 0:
        seqInfo = readFile(seq, aug) 
        writeSeq(seq, seqInfo)
    elif numShifts == 1:
        seqInfo = readFile(seq, aug) 
        newSeqInfo = {}

        for turnComb in seqInfo:
            turns = turnComb.split()
            turnType1 = turns[0][:len(turns[0]) - 2]
            turnNum1 = int(turns[0][len(turns[0]) - 1])
            turnType2 = turns[1][:len(turns[1]) - 2]
            turnNum2 = int(turns[1][len(turns[1]) - 1])
            
            turnNum2 = turnNum2 + 1
            turnNum1 = turnNum1 + 1
            newTurnComb = turnType1 + "_" + str(turnNum1) + " " + turnType2 + "_" + str(turnNum2)

            if (turnNum2 == 6):
                turnNum2 = 3
                turnNum1 = 0
                newTurnComb = turnType2 + "_" + str(turnNum1) + " " + turnType1\
                + "_" + str(turnNum2) #NOT A WEIRD TYP0 weird switch happens

            newSeqInfo[newTurnComb] = seqInfo[turnComb]

        newSeq = seq[5] + seq[0:5]
        writeSeq(newSeq, newSeqInfo)

    else:
        print "Not yet implemented\n"

def writeSeq(seq, seqInfo):
    newFile = open("data/Out_population_" + seq + ".txt", "w")
    newFile.write('%6s %6s %10s %10s\n' % ('turn 1', 'turn 2','counts', 'population'))

    for turnComb in seqInfo:
        turns = turnComb.split()
        newFile.write('%6s %6s %10s %10s\n' % (turns[0], turns[1],'0',str(seqInfo[turnComb])))
    newFile.close()

def readFile(seq, aug):
    popFile = ''
    if aug:
        popFile = open("data/Out_population_" + seq + ".txt", "r")
    else:
        popFile = open("rawData/Out_population_" + seq + ".txt", "r")

    seqInfo = {}
    for line in popFile:
        words = line.split()
        if words[0] != "turn":
            turnComb = words[0] + " " + words[1]
            seqInfo[turnComb] = float(words[3])
    popFile.close()

    return seqInfo

allFiles = os.listdir("rawData")

for name in allFiles:
    seq = name[15:21]
    if seq[0:3] == "AAA":
        shiftLeft(seq, 3, False)
    else:
        shiftRight(seq, 0, False)

allRefinedFilesOne = os.listdir("data")

for name in allRefinedFilesOne:
    seq = name[15:21]
    shiftRight(seq, 1, True)
    shiftLeft(seq, 1, True)

allRefinedFilesTwo = os.listdir("data")
refinedFiles = []
for filename in allRefinedFilesTwo:
    if filename not in allRefinedFilesOne:
        refinedFiles.append(filename)

for name in refinedFiles:
    seq = name[15:21]
    shiftRight(seq, 1, True)
    shiftLeft(seq, 1, True)

allRefinedFilesThree = os.listdir("data")
refinedFiles = []
for filename in allRefinedFilesThree:
    if filename not in allRefinedFilesTwo:
        refinedFiles.append(filename)

for name in refinedFiles:
    seq = name[15:21]
    shiftRight(seq, 1, True)
