import numpy as np
import matplotlib.pyplot as plt
import math


# importing warnings to avoid system warning
import warnings
warnings.simplefilter("ignore")

def main():
    print('START Q3_C\n')
	
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
    
    # load data 
    Feturs = []
    for IdxDummy in range(len(Cont3)):
        dummyData = Cont3[IdxDummy]
        Feturs.append([dummyData[0], dummyData[1], dummyData[2]])
    Label_v = []
    for IdxDummy in range(len(Cont3)):
        dummyData = Cont3[IdxDummy]
        Label_v.append(0 if dummyData[3]=='M' else 1)
    Feturs = np.array(Feturs)
    Label_v = np.array(Label_v)

    # run leave one out module
    Instances1 = LOOMain(Feturs, Label_v)
    Acc1 = Instances1.PrepareLeaveOneOut()
    Acc1 = round(Acc1, 2)
    print("Leave One Out ")
    print("Input -> Height, Weight and Age")
    print("Accuracy ->",Acc1)
    print('END Q3_C\n')

# class for logistic regression
class LogidticRegressionClassificationClass:
    
    # default function
    def __init__(self,alphaValue=0.001,iterTotal=400):
        self.alphaValue = alphaValue
        self.iterTotal = iterTotal
    
    # training function
    def StartTraining(self,XComp,yValDummy):
        shapeweightSize = (np.shape(XComp)[1]+1,1)
        weightsCalcu = np.zeros(shapeweightSize)
        shapeFeatur = (np.shape(XComp)[0],1)
        XComp = np.c_[np.ones(shapeFeatur),XComp]
        self.costlistTotal = np.zeros(self.iterTotal,)
        for iItrCont in range(self.iterTotal):
            zProduct = np.dot(XComp,weightsCalcu)
            
            # weight calculations
            weightsCalcu = weightsCalcu - self.alphaValue*np.dot(XComp.T,(1/(1+np.exp(-zProduct)))-np.reshape(yValDummy,(len(yValDummy),1)))
            theta = weightsCalcu
            z0Calcu = np.dot(XComp,theta)
            calOneDummy = (1+np.exp(-z0Calcu))
            cost0Dummy = yValDummy.T.dot(np.log(1/calOneDummy))
            caltwoDummy = (1+np.exp(-z0Calcu))
            cost1Check = (1-yValDummy).T.dot(np.log(1-(1/caltwoDummy)))
            costTotalsum = (cost1Check + cost0Dummy)
            costFinalval = -(costTotalsum)/len(yValDummy)
            self.costlistTotal[iItrCont] = costFinalval
        self.weightsCalcu = weightsCalcu
    
    # predict function for prediction
    def predictouts(self,XComp):
        weightsInstance = np.zeros((np.shape(XComp)[1]+1,1))
        Xins = np.c_[np.ones((np.shape(XComp)[0],1)),XComp]
        Answer = weightsInstance,Xins
        zProduct = np.dot(Answer[1],self.weightsCalcu)
        lis1 = [1 if iItrCont>0.5 else 0 for iItrCont in (1/(1+np.exp(-zProduct)))]
        return lis1

# class for leave one out system 
class LOOMain:
    
    # constructor
    def __init__(self, TrainTempData8, TrainLabelData7):
        self.TrainTempData8 = TrainTempData8
        self.TrainLabelData7 = TrainLabelData7
        
    # main run of the code
    def PrepareLeaveOneOut(self):
        TrainTempData8 = self.TrainTempData8
        TrainLabelData7 = self.TrainLabelData7
        PredictionOutsList = []
        for LoopItration in range(len(TrainTempData8)):
            InstanceFeatureList3 = [ list(yValDummy) for yValDummy in list(TrainTempData8)]
            InstanceLabelList3 = list(TrainLabelData7)
            
            # dividing data
            Test_Feature = InstanceFeatureList3[LoopItration]
            Test_Label = InstanceLabelList3[LoopItration]
            del InstanceFeatureList3[LoopItration]
            del InstanceLabelList3[LoopItration]
            
            # loading
            SampleFeatures0 = np.array(InstanceFeatureList3)
            SampleLabels0 = np.array(InstanceLabelList3)
            Target_Features = np.array([Test_Feature])
            
            # prediction
            TempRunData = LogidticRegressionClassificationClass(alphaValue=0.01,iterTotal=1000)
            TempRunData.StartTraining(SampleFeatures0, SampleLabels0)
            DummyPredictions = TempRunData.predictouts(Target_Features)
            PredictionOutsList.append([Test_Label, DummyPredictions[0]])
        
        # comparison
        out1 = [float(xValDummy[0]) for xValDummy in PredictionOutsList]
        out2 = [float(xValDummy[1]) for xValDummy in PredictionOutsList]
        corr = 0
        for iItrCont in range(len(out1)):
            if out1[iItrCont] == out2[iItrCont]:
                corr += 1
        return corr / float(len(out2)) * 100.0

if __name__ == "__main__":
    main()
    