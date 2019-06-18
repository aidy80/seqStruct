import os

turnDict = {}

for filename in os.listdir("rawData"):
    currfile = open("rawData/" + filename, "r")
    for line in currfile:
        words = line.split()
        if words[0] != "turn":
            turnComb = words[0][:-2] + " " + words[1][:-2]
            if turnComb in turnDict.keys():
                turnDict[turnComb] += float(words[3])
            else:
                turnDict[turnComb] = 0

totalCounts = 0
for turnComb in turnDict.keys():
    totalCounts += turnDict[turnComb]

for turnComb in turnDict.keys():
    turnDict[turnComb] = float(turnDict[turnComb]) / float(totalCounts)
    print turnComb, turnDict[turnComb]


print len(turnDict)
