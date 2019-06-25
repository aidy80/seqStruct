import os
import math

totEnt = 0.0
numFiles = 0

for filename in os.listdir("data"):
    popFile = open("data/" + filename, "r")
    ent = 0
    for line in popFile:
        words = line.split()
        if words[0] != "turn":
            ent += - float(words[3])/100.0 * math.log10(float(words[3])/100.0)

    totEnt += ent
    numFiles += 1

print "totAvgEnt", totEnt / float(numFiles)
