import sklearn
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import SimpleImputer

import parseData
import os
import numpy as np

def numDiff(seq1, seq2):
    numD = 0
    for index, letter in enumerate(seq1):
        if letter != seq2[index]:
            numD += 1
    return numD

def numAla(seq):
    numAla = 0
    for letter in seq:
        if letter == 'A':
            numAla += 1
    return numAla

def convertToPercent(pred):
    popSum = 0.0
    for turnKey in pred:
        popSum += pred[turnKey]

    normFactor = 100.0 / popSum
    for turnKey in pred:
        pred[turnKey] *= normFactor

    return pred


class predictPop():
    def __init__(self): 
        self.allPops, self.allTurnCombs = parseData.getSeqInfo3A()
        self.aminos = parseData.getAminos()
        self.numX = 6
        self.trainInsts = []
        self.trainLabels = []
        self.trainWeights = []
        self.testInsts = []
        self.testLabels = []
        self.energy = parseData.getEnergyInfo()

    def genData(self, turnComb, testSeqs):
        turnPops = parseData.getTurnCombPops(self.allPops, turnComb)
        for seq in turnPops:
            normSeq = self.genSeqVec(seq)
            fullVec = normSeq
            
            if turnComb != "other":
                energyVec = self.genEnergyVec(seq, turnComb)
                fullVec = np.concatenate((normSeq, energyVec), axis=None)

            if seq in testSeqs:
                self.testInsts.append(fullVec)
                self.testLabels.append(turnPops[seq])
            else: 
                self.trainInsts.append(fullVec)
                self.trainLabels.append(turnPops[seq])
                totNumDiff = 0
                for testSeq in testSeqs:
                    totNumDiff += numDiff(testSeq, seq)
                avgNumDiff = totNumDiff / float(len(testSeqs))
                if avgNumDiff >= 5:
                    weight = 0
                elif avgNumDiff >= 4:
                    weight = 1
                elif avgNumDiff >= 3:
                    weight = 3
                elif avgNumDiff >= 2:
                    weight = 7
                else:
                    weight = 15

                #weight = 1

                self.trainWeights.append(weight)

    def resetTrainTest(self):
        self.trainInsts = []
        self.trainLabels = []
        self.testInsts = []
        self.testLabels = []
        self.trainWeights = []

    def predictAllTurns(self, testSeqs):
        #mseAvg = 0.0
        for testSeq in testSeqs:
            if os.path.isfile("coef/"+testSeq):
                os.remove("coef/"+testSeq)

            pred = {}
            for turnComb in self.allTurnCombs:
                pred[turnComb] = 0

            for turnComb in self.allTurnCombs:
                self.resetTrainTest()
                self.genData(turnComb, testSeq)
                pred[turnComb] = self.predictTurnPop(testSeq, turnComb)

            for turnComb in pred:
                if pred[turnComb] < 0:
                    pred[turnComb] = 0

            pred = convertToPercent(pred)

            self.outputPred(pred, testSeq)
            #mseAvg += self.mse(pred,testSeq)

        #return mseAvg / float(len(testSeqs))

    
    def outputPred(self, pred, seq):
        predFile = open("predictions/Out_prediction_" + seq, "w") 
        predFile.write('%10s %10s %10s\n' % ('turn comb','pred', 'true'))
        for turnComb in pred:
            if turnComb in self.allPops[seq]:
                predFile.write('%10s %10.6f %10.6f\n' % (turnComb, \
                    pred[turnComb],self.allPops[seq][turnComb]))
            else:
                predFile.write('%10s %10.6f %10.6f\n' % (turnComb, \
                    pred[turnComb], 0.0))

        predFile.write("mse: %s" % (self.mse(pred, seq)))
        predFile.close()

    def mse(self, pred, seq):
        trueTurns = self.allPops[seq]
        mse = 0.0
        for turn in pred:
            if turn in trueTurns:
                mse += (trueTurns[turn] - pred[turn])**2
            else:
                mse += pred[turn]**2
        mse /= len(pred)

        return mse

    def normalizePred(self, pred):
        popSum = 0.0
        for key in pred:
            popSum += pred[key]

        normFactor = 100.0 / popSum

        for key in pred:
            pred[key] = pred[key] * normFactor

        return pred
         
    def predictTurnPop(self, testSeq, turnComb):
        #reg = linear_model.Ridge(alpha=1.0)
        reg = SVR(kernel = 'rbf', C=1.0, gamma=0.05)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(self.trainInsts)
        reg.fit(imp.transform(self.trainInsts), self.trainLabels, \
                sample_weight=np.array(self.trainWeights))
        #reg.fit(self.trainInsts, self.trainLabels, \
        #        sample_weight=np.array(self.trainWeights))
        #self.printCoef(testSeq, turnComb, reg)

        normSeq = self.genSeqVec(testSeq)
        fullVec = normSeq
        if turnComb != "other":
            energyVec = self.genEnergyVec(testSeq, turnComb)
            fullVec = np.concatenate((normSeq, energyVec), axis=None)
            fullVec = fullVec.reshape(1, -1)
            fullVec = imp.transform(fullVec)
        else:
            fullVec = fullVec.reshape(1, -1)
        return reg.predict(fullVec)

    def printCoef(self, seq, turnComb, reg):
        coefFile = open("coef/" + seq, "a+")
    
        numPrinted = 0
        coefFile.write(turnComb + "\n")
        for coef in reg.coef_:
            if numPrinted >= len(self.aminos):
                numPrinted = 0 
                coefFile.write("\n")
            coefFile.write(str(coef) + " ")
            numPrinted+=1

        coefFile.write("\n")
        coefFile.close()

    def genEnergyVec(self, seq, turnComb):
        energyVec = np.zeros(len(self.energy["AAADSV"]["II_0II_3"]))
        #energyVec = np.zeros(1)
        if turnComb in self.energy[seq]:
            for index, energy in enumerate(self.energy[seq][turnComb]):
                energyVec[index] = self.energy[seq][turnComb][energy]
                #if energy == "PE":
                #    energyVec[0] = self.energy[seq][turnComb][energy]
        else:
            for index, energy in enumerate(energyVec):
                energyVec[index] = np.nan
                print seq, turnComb, "Nanning"
                #if energy == "PE":
                #    energyVec[0] = self.energy[seq][turnComb][energy]

        return energyVec

    def genSeqVec(self, seq):
        inv_aminos_map = {v: k for k, v in self.aminos.iteritems()} 
        seq_vec = np.zeros(len(self.aminos) * self.numX)
        for index, amino in enumerate(seq):
            if (index < self.numX):
                seq_vec[inv_aminos_map[amino] + index*len(self.aminos)] = 1

        return seq_vec 

    """
    def genSeqVec(self, seq):
        inv_aminos_map = {v: k for k, v in self.aminos.iteritems()} 
        seq_vec = np.zeros(len(self.aminos)**2 * self.numX)
        for index, amino in enumerate(seq):
            if (index < self.numX):
                #print amino, seq[(index + 1) % 6], len(self.aminos) * inv_aminos_map[amino] + \
                #        inv_aminos_map[seq[(index+1)% self.numX]] + \
                index*len(self.aminos)**2
                seq_vec[len(self.aminos) * inv_aminos_map[amino] + \
                        inv_aminos_map[seq[(index+1)%self.numX]] + \
                index*len(self.aminos)**2] = 1

        return seq_vec 
    """

    def testEn(self):
        parameters = {'n_estimators' : [1,5, 10, 50], 'learning_rate' : [0.01,
            0.5,0.75], 'loss' : ['square']}
        reg = linear_model.Ridge(alpha=1.0)
        enReg = AdaBoostRegressor(reg)

        clf = GridSearchCV(enReg, parameters, cv=10, scoring='neg_mean_squared_error')
        clf.fit(self.trainInsts, self.trainLabels)
        print clf.cv_results_['mean_test_score']

    def testSVR(self):
        #Kernals/Lambdas tested using SVC
        parameters = {'C' : [0.001, 0.1, 1, 10, 1000],
        'kernel':['rbf']}
        
        #Test several different hyperparameters
        svr = SVR()
        clf = GridSearchCV(svr, parameters, cv=10, scoring='neg_mean_squared_error')
        clf.fit(self.trainInsts, self.trainLabels)
        print clf.cv_results_['mean_test_score']

    def testRidge(self):
        #Kernals/Lambdas tested using SVC
        parameters = {'alpha' : [0.1, 0.5, 1.0, 10, 100], 'normalize' : [True,\
                False], 'kernel' : ['linear', 'rbf']}

        #Test several different hyperparameters
        reg = linear_model.Ridge()
        clf = GridSearchCV(reg, parameters, cv=100,\
                scoring='neg_mean_squared_error', return_train_score=True)
        clf.fit(self.trainInsts, self.trainLabels)
        print clf.cv_results_['mean_train_score']

    def testLasso(self):
        #Kernals/Lambdas tested using SVC
        parameters = {'alpha' : [0.1, 0.5, 1.0, 10, 100]}
        
        #Test several different hyperparameters
        reg = linear_model.Lasso()
        clf = GridSearchCV(reg, parameters, cv=10, scoring='neg_mean_squared_error')
        clf.fit(self.trainInsts, self.trainLabels)
        print clf.cv_results_['mean_test_score']


ex = predictPop()
#ex.predictAllTurns(["FRSANR"])

allSeqInfo, allTurnCombs = parseData.getSeqInfo3A()

doneSeqs = []

for seq in allSeqInfo:
    if numAla(seq) < 3 and seq not in doneSeqs:
        print seq
        miniSeq = [seq]
        ex.predictAllTurns(miniSeq)

        doneSeqs.append(seq)
        currSeq = seq
        for i in range(5):
            currSeq = currSeq[5] + currSeq[0:5]
            doneSeqs.append(currSeq)


