'''
Author: Ashish Katlam
Description: This code trains the malware detection model

'''

import pickle
import datetime
import sys
from sklearn import svm
import random
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time
import rbf_print as rp

def load_data():
    feature_vector = []
    # Data format
    # Column 0: Class label (1: Malware, 0: Benign)
    # Column 1-19: Features
    with open("final_data.csv",'r') as fp:
        for i,line in enumerate(fp):
            if i == 0:
                pass
            else:
                feature_vector.append([int(x.strip()) for x in line.split(',')])
    return feature_vector


'''
Driver Function
Usage: python model_train.py train_test/k-fold [roc]
'''
if __name__ == "__main__":

    print 'Start to training'
    # Check for arguments
    if len(sys.argv) <= 1:
        # Load the data
        data = load_data()

        # Shuffle the data
        random.shuffle(data)

        # Divide the data into training and testing in 60:40
        trainLength = int(0.7*len(data))

        # Training Data
        trainX = [x[:-1] for x in data[:trainLength]]
        trainY = [y[-1] for y in data[:trainLength]]

        # Testing Data
        testX = np.array([x[:-1] for x in data[trainLength:]])
        testY = np.array([y[-1] for y in data[trainLength:]])

        #MLPClassifier
        clf_mlp = MLPClassifier()
        #save clf to plot roc_curve
        #clf_list = [clf_knn,clf_adaboost,clf_nb,clf_rf,clf_lgbm,clf_svm]
        clf_list = [clf_mlp]
        #clf_labels = ['knn','ada','nb','rf','lgbm','svm']
        clf_labels = ['mlp']
        
        rp.log('*'*50)
        # Perform training
        index = 0
        for clf in clf_list:
            rp.log('[{}] Training the data....'.format(clf_labels[index]))
            clf.fit(np.array(trainX), np.array(trainY))
            predicted = clf.predict(testX)
            #rp.log('[{}] default Accuracy: {:.3f}%'.format(clf_labels[index],clf.score(testX, testY)*100))
            #rp.log('[{}] f1-score: {:.3f}%'.format(clf_labels[index],metrics.f1_score(testY,predicted)*100))
            #rp.log(metrics.classification_report(testY,predicted))
            print('[{}] default Accuracy: {:.3f}%'.format(clf_labels[index],clf.score(testX, testY)*100))
            print(metrics.classification_report(testY,predicted))
            index += 1

        #rp.log('*'*30)
                

    else:
        print '[+] Usage: python {}>'.format(__file__)
    

