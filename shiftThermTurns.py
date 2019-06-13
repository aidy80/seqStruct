#Aidan Fike
#June 12, 2019

#Program to create cyclic equivalents for all of the energy information found
#through Be-Meta

import os
import shutil

numAminos = 6

def shiftRight(seq, fromDir, toDir):
    print "currSeq=", seq
    fromFiles = os.listdir("energy/" + fromDir + "/" + seq)
    newSeq = seq[5] + seq[0:5]
    if not os.path.exists("energy/" + toDir + "/" + newSeq):
        os.mkdir("energy/" + toDir + "/" + newSeq)
    for filename in fromFiles:
        turnComb = filename[5:].split("+")

        turnType1 = turnComb[0] 
        turnNum1 = int(turnType1[len(turnType1) - 1])
        turnType1 = turnComb[0][:-2] 

        turnType2 = turnComb[1].split(".")[0]
        turnNum2 = int(turnType2[len(turnType2) - 1])
        turnType2 = turnComb[1].split(".")[0][:-2] 

        turnNum2 = turnNum2 + 1
        turnNum1 = turnNum1 + 1
        newTurnComb = turnType1 + "_" + str(turnNum1) + "+" + turnType2 + "_" + str(turnNum2)

        if (turnNum2 == 6):
            turnNum2 = 3
            turnNum1 = 0
            newTurnComb = turnType2 + "_" + str(turnNum1) + "+" + turnType1\
            + "_" + str(turnNum2) #NOT A WEIRD TYP0 weird switch happens

        shutil.copyfile("energy/" + fromDir + "/" + seq + "/" + filename, \
                 "energy/" + toDir + "/" + newSeq + "/enth_" + newTurnComb + ".dat") 
        if os.path.exists("energy/allEnergy/"+newSeq):
            shutil.rmtree("energy/allEnergy/"+newSeq)
        shutil.copytree("energy/" + toDir + "/" + newSeq, "energy/allEnergy/"+\
                                                                    newSeq)

thermDirs = os.listdir("energy/OG")

for thermDir in thermDirs:
    shiftRight(thermDir, "OG", "shift1")

for i in range(1, numAminos):
    print "currShift=", i
    thermDirs = os.listdir("energy/shift" + str(i))

    for thermDir in thermDirs:
        shiftRight(thermDir, "shift" + str(i), "shift" + str(i+1))
