import numpy as np
import matplotlib.pyplot as plt
import math

# importing warnings to avoid system warning
import warnings
warnings.simplefilter("ignore")

def main():
    print('START Q3_AB\n')
	
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
    featuresListdataNew = np.array([x[1:-1] for x in Cont3])
    labelListdataNew = np.array([0 if x[-1]=='M' else 1 for x in Cont3])

    # model run
    trainedRunningModel = LogidticRegressionClassificationClass(alphaValue=0.001,iterTotal=400000)
    trainedRunningModel.StartTraining(featuresListdataNew, labelListdataNew)
    dataForAxisOne = [x[0] for x in featuresListdataNew]
    dataForAxisTwo = [x[1] for x in featuresListdataNew]
    completeNewFeaturesdata = []
    for x in np.arange(int(min(dataForAxisOne)-1)-5.0,int(max(dataForAxisOne)+1)+5.0,0.1):
        for y in np.arange(int(min(dataForAxisTwo)-1)-5.0,int(max(dataForAxisTwo)+1)+5.0,0.1):
            completeNewFeaturesdata.append([x,y])
    completeNewFeaturesdata = np.array(completeNewFeaturesdata)
    predictedoriginaldata = trainedRunningModel.predictouts(featuresListdataNew)
    predictedvalData = trainedRunningModel.predictouts(completeNewFeaturesdata)

    # plotting graph 
    plt.scatter([ completeNewFeaturesdata[x][0] for x in range(len(completeNewFeaturesdata)) if predictedvalData[x]==1  ], [ completeNewFeaturesdata[x][1] for x in range(len(completeNewFeaturesdata)) if predictedvalData[x]==1  ], c='lightcyan' )
    plt.scatter([ completeNewFeaturesdata[x][0] for x in range(len(completeNewFeaturesdata)) if predictedvalData[x]==0  ], [ completeNewFeaturesdata[x][1] for x in range(len(completeNewFeaturesdata)) if predictedvalData[x]==0 ], c='lightcoral'   )
    plt.scatter([ featuresListdataNew[x][0] for x in range(len(featuresListdataNew)) if predictedoriginaldata[x]==1  ], [ featuresListdataNew[x][1] for x in range(len(featuresListdataNew)) if predictedoriginaldata[x]==1 ], c='red', marker='<' )
    plt.scatter([ featuresListdataNew[x][0] for x in range(len(featuresListdataNew)) if predictedoriginaldata[x]==0  ], [ featuresListdataNew[x][1] for x in range(len(featuresListdataNew)) if predictedoriginaldata[x]==0 ], c='blue' , marker='>'  )
    plt.title("Prediction Plot")
    plt.show()
    print('END Q3_AB\n')

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
