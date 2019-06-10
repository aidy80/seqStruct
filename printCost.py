import matplotlib.pyplot as plt

costFile = open("costs/cost", "r")
mtestFile = open("costs/mTest", "r")
mtrainFile = open("costs/mTrain", "r")
costs = []
times = []
mtest = []
mtrain = []
for line in costFile:
    words = line.split()
    times.append(float(words[0]))
    costs.append(float(words[1]))
for line in mtestFile:
    words = line.split()
    mtest.append(float(words[1]))
for line in mtrainFile:
    words = line.split()
    mtrain.append(float(words[1]))

print len(costs), len(mtest)
mtest.pop()
mtrain.pop()
mTimes = times[0:len(times):len(times)/len(mtest)]
fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')

ax[0].scatter(times,costs)
ax[1].scatter(mTimes,mtest)
ax[2].scatter(mTimes,mtrain)
plt.xlim(0,len(times))

fig.savefig("costs/all.png")
plt.show()

costFile.close()
mtestFile.close()
mtrainFile.close()