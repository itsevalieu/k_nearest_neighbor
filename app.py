import numpy as np
import csv
from sklearn.metrics import confusion_matrix

def knearestneighbor(trainfile, testfile, outputfile, k=3):
    print(trainfile, testfile, outputfile, k)
    with open(outputfile, 'w') as f:
        Xtrain, Ytrain = read_trainfile(trainfile)
        Xtest, Ytest = read_testfile(testfile)

        #create numpy array Yhat(1x20) consisting of all zeros
        y_hat = np.zeros(shape=(1, 20))

        #for each example in Xtest, apply the k-nearest neighhbor method to determine class
        for i in range(len(y_hat[0])):
            d = find_distance(Xtrain, Xtest, i)
            idx = np.argpartition(d, k)
            knn = d[idx[0]] + d[idx[1]] + d[idx[2]]
            yhat_class = populate_yhat(knn)

            # put result of knn method in Yhat
            y_hat[0][i] = yhat_class

        for line, point in enumerate(Ytest[0]):
            f.write(str(Ytest[0][line]) + "," + str(y_hat[0][line])+ "\n")

        # using scikit-learn to find confusion matrix
        confusionmatrix = confusion_matrix(Ytest[0], y_hat[0])
        print(confusionmatrix)
        # receiving index error when trying to use just numpy to find confusionmatrix. Why?
            # confusionmatrix = np.zeros(shape=(2, 2))
            # actual = np.array(Ytest[0])
            # predicted = np.array(y_hat[0])
            # for a, b in zip(actual, predicted):
            #     print(a,b)
            #     confusionmatrix[a][b] += 1

    return confusionmatrix
def populate_yhat(knn):
    if knn > 1:
        knn_class = 1
        return knn_class
    else:
        knn_class = 0
        return knn_class
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
def find_distance(x_train, x_test, test_point=0):
    # first subtract and ^2 each train array with a single test point, then append to d
    d = []
    for i in range(len(x_test)):
        dist = (x_train[i][:] - x_test[i][test_point])**2
        d.append(dist)

    # add all points together across lists
    sum = 0
    for i, j in enumerate(d):
        sum += d[i][:]

    # sqrt each sum to find distance for all points, append to distance and return
    distance = []
    for m in sum:
        n = np.sqrt(m)
        distance.append(n)
    return distance


knearestneighbor('iristrain.csv', 'iristest.csv', 'irisoutput', 3)
