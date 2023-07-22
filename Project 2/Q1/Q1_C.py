import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    print('START Q1_C\n')
	
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
    
    # Loops for K values
    for kItr in range(1,11):
        ListoutCollection = []
        
        # Loop for D values
        for dItr in range(0,7):
            Outlist = []
            for tmpdataVal in  Cont1:
                
                # get unit value
                xValDummy = tmpdataVal[0]
                
                # prediction
                instanceList1 = [ math.sin(di*kItr*xValDummy)*math.sin(di*kItr*xValDummy) for di in range(1, dItr+1)]
                yValDummy = 1  + sum(instanceList1)
                Outlist.append(yValDummy)
            Outlist = np.array(Outlist)
            ListoutCollection.append(Outlist)
        
        # Plotting all data points
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[0], c='r', label = " d=0")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[1], c='g', label = " d=1")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[2], c='b', label = " d=2")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[3], c='c', label = " d=3")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[4], c='m', label = " d=4")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[5], c='y', label = " d=5")
        plt.scatter([tmpdataVal[0] for tmpdataVal in  Cont1], ListoutCollection[6], c='k', label = " d=6")
        plt.title("K value "+str(kItr))
        plt.legend()
        plt.show()


    print('END Q1_C\n')


if __name__ == "__main__":
    main()
    