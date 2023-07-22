import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    print('START Q2_D\n')
	
    # Getting data from file 1
    Cont1 = []
    Cont1Alldata = open("datasets/Q1_B_train.txt","r").readlines()
    for IdxDummy in range(len(Cont1Alldata)):
        DummyCntnt = Cont1Alldata[IdxDummy].replace(" ","").replace("\n","").replace("(","").replace(")","").split(",")
        DummyCntnt = [float(DummyCntnt[0]), float(DummyCntnt[1])]
        Cont1.append(DummyCntnt)

    # Getting data from file 2
    Cont2 = []
    Cont2Alldata = open("datasets/Q1_C_test.txt","r").readlines()
    for IdxDummy in range(len(Cont2Alldata)):
        DummyCntnt = Cont2Alldata[IdxDummy].replace(" ","").replace("\n","").replace("(","").replace(")","").split(",")
        DummyCntnt = [float(DummyCntnt[0]), float(DummyCntnt[1])]
        Cont2.append(DummyCntnt)

    # Getting data from file 3
    Cont3 = []
    Cont3Alldata = open("datasets/Q3_data.txt","r").readlines()
    for IdxDummy in range(len(Cont3Alldata)):
        DummyCntnt = Cont3Alldata[IdxDummy].replace(" ","").replace("\n","").replace("(","").replace(")","").split(",")
        DummyCntnt = [float(DummyCntnt[0]), float(DummyCntnt[1]), int(DummyCntnt[2]), str(DummyCntnt[3])]
        Cont3.append(DummyCntnt)
    
    # loading data 
    XTrainLeftSide = np.array([xValDummy[0] for xValDummy in Cont1[:20]])
    Y_Train = np.array([xValDummy[1] for xValDummy in Cont1[:20]])
    X_Test = np.array([xValDummy[0] for xValDummy in Cont2])
    YTestRightSide = np.array([xValDummy[1] for xValDummy in Cont2])

    # model run
    object1 = localRegressionWght()
    prediction = [object1.StartTraining(x0InitilizationVal, XTrainLeftSide, Y_Train) for x0InitilizationVal in X_Test]
    CalmseError = ((prediction - YTestRightSide)**2).mean()
    object1.displayPrintContentInfo(XTrainLeftSide, YTestRightSide, CalmseError)
    object1.plot(X_Test, prediction)
    
    print('END Q2_D\n')

# class for local weighted regressor 
class localRegressionWght: 
    
    # constructor
    def __init__(self):
        self.tauValCal=0.204
    
    # printing function
    def displayPrintContentInfo(self, XTrainLeftSide, YTestRightSide, CalmseError):
        print("Training Data Size -", len(XTrainLeftSide))
        print("Testing Data Size -", len(YTestRightSide))
        print("MSE -", CalmseError)
    
    # plotting function
    def plot(self, XComp, YComp):
        plt.title("Locally Weighted Linear Regression")
        plt.scatter(XComp, YComp, c='b')
        plt.show()
    
    # training function
    def StartTraining(self, x0InitilizationVal, XComp, YComp):
        x0InitilizationVal = np.r_[1, x0InitilizationVal]
        XComp = np.c_[np.ones(len(XComp)), XComp]
        xwghtVal = XComp.T * np.exp(np.sum((XComp - x0InitilizationVal) ** 2, axis=1) / (-2 * (self.tauValCal **2) ))
        return x0InitilizationVal @ np.linalg.pinv(xwghtVal @ XComp) @ xwghtVal @ YComp

if __name__ == "__main__":
    main()
    