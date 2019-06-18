#Aidan Fike
#June 12, 2019

#Program to plot the costs, training, and testing accuracies to a photo form
#and onto terminal. Pass in the testNum and batchNum as a command line
#argument. For example,

#python printCost 1.2

import matplotlib.pyplot as plt
import sys

costFile = open("costs/cost" + str(sys.argv[1]), "r")
mtestFile = open("costs/mTest" + str(sys.argv[1]), "r")
mtrainFile = open("costs/mTrain" + str(sys.argv[1]), "r")

costs = []
times = []
mtest = []
mtrain = []
metric = "-1"

for line in costFile:
    words = line.split()
    times.append(float(words[0]))
    costs.append(float(words[1]))

first = True
for line in mtestFile:
    words = line.split()
    if first:
        metric = words[0]
        first = False
    mtest.append(float(words[1]))

first = True
for line in mtrainFile:
    words = line.split()
    if first:
        metric = words[0]
        first = False
    mtrain.append(float(words[1]))

fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

ax[0].scatter(times,costs)
ax[0].set_ylabel("log liklihood cost")
ax[1].scatter(times,mtest)
ax[1].set_ylabel("test " + metric)
ax[2].scatter(times,mtrain)
ax[2].set_ylabel("train " + metric)
ax[2].set_xlabel("epochs")
#plt.xlim(0,len(times))

fig.savefig("costs/fig" + str(sys.argv[1]) + ".png")
plt.show()

costFile.close()
mtestFile.close()
mtrainFile.close()
