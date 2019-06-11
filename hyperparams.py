class params():
    learning_rate = 0.001
    n_epochs = 12000
    regBeta=0.0

    batchSize = 10

    lr_decay=0.99
    decay_epoch=1000
    nImproveTime=2000
    pearBail = 0.013
    lrThresh=1e-5
    largeDecay=0.1
    calcMetStep = 100
    earlyStop=True
    saveBest=False

    numX = 6

    numHiddenNodes = 0
    numCLayers = 3
    numChannels = [64]*numCLayers
    filterHeights = [2,3]
    keep_prob=[0.6]*(numCLayers+1)
    leakSlope = 0.01


