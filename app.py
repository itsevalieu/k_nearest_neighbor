import numpy as np
import csv

def knearestneighbor(trainfile, testfile, outputfile, k):
    print(trainfile, testfile, outputfile, k)
    Xtrain, Ytrain = read_trainfile(trainfile)

    Xtest, Ytest = read_testfile(testfile)

    #create numpy array Yhat(1x20) consisting of all zeros
    y_hat = np.zeros(shape=(1, 20))

    #for each example in Xtest, apply the k-nearest neighhbor method to determine class
        # put result of knn method in Yhat
    ## write_outputfile(outputfile)

    ## confusionmatrix = [] #should contain 4 values slide for confusion matrix
    ## return confusionmatrix

def read_trainfile(trainfile):
    #read trainfile
    with open(trainfile, 'r') as iristrain:
        reader = csv.reader(iristrain, delimiter=",")
        train_list = []
        next(reader)
        for i, line in enumerate(reader):
            lsepal, wsepal, lpetal, wpetal, species = line
            train_list.append([float(lsepal), float(wsepal), float(lpetal), float(wpetal), float(species)])

        #populate numpy array X(4 x 80) and numpy array Y(1 x 80) slide 23 linear regression
        x = np.zeros(shape=(4, 80))
        y = np.zeros(shape=(1, 80))

        for j, line in enumerate(train_list):
            x[0][j] = line[0]
            x[1][j] = line[1]
            x[2][j] = line[2]
            x[3][j] = line[3]
            y[0][j] = line[4]
        return x, y

def read_testfile(testfile):
    # read testfile
    with open(testfile, 'r') as iristest:
        reader = csv.reader(iristest, delimiter=",")
        test_list = []
        next(reader)
        for i, line in enumerate(reader):
            lsepal, wsepal, lpetal, wpetal, species = line
            test_list.append([float(lsepal), float(wsepal), float(lpetal), float(wpetal), float(species)])

        # populate numpy array Xtext (4x20) and numpy array Ytest(1x20)
        x = np.zeros(shape=(4, 20))
        y = np.zeros(shape=(1, 20))

        for j, line in enumerate(test_list):
            x[0][j] = line[0]
            x[1][j] = line[1]
            x[2][j] = line[2]
            x[3][j] = line[3]
            y[0][j] = line[4]
        return x, y

def write_outputfile(outputfile):
    # write to outputfile
    with open(outputfile, 'w'):
        print(outputfile)


knearestneighbor('iristrain.csv', 'iristest.csv', 'irisoutput', 3)