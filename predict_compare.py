'''
Author: Bingfei Ren 
Description: This code classifies given apks into Malware/Benign using malware-override Voting for SVM ADABoosting RandormForest LightGBM
'''
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import pickle,cPickle
import sys
import numpy as np
import random

def load_data():
    feature_vector = []
    #Data format
    with open('final_data.csv','r') as fp:
        for i,line in enumerate(fp):
            if i==0:
                pass
            else:
                feature_vector.append([int(x.strip()) for x in line.split(',')])
        return feature_vector

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        
        # Extract the features for a given application
        #features = extract_features(file_path)
        # Form the feature vector
        #feature_vector = create_vector_single(features)
        # Load the pre configured feature model from a pickle file
        #model = pickle.load(open("feature_model.p", "rb"))
        # Reduce the feature vector into size 12
        #feature_vector_new = model.transform(feature_vector)

        data = load_data()
        random.shuffle(data)
        #using 10% to test the precise like model_match used.
        testLength = int(0.9*len(data))
        testX = [x[:-1] for x in data[testLength:]]
        testY = [y[-1] for y in data[testLength:]]
        
        # Load the pre-trained model from a pickle file
        clf_svm = cPickle.load( open( "train_data_svm.p", "rb" ) )
        clf_ada = cPickle.load( open( "train_data_ada.p", "rb") )
        clf_rf = cPickle.load( open( "train_data_rf.p", "rb") )
        clf_lgbm = cPickle.load( open( "train_data_lgbm.p", "rb") )
        #clf_mlp = cPickle.load(open("train_data_mlp.p","rb"))
        clf_knn = cPickle.load(open("train_data_knn.p","rb"))

        right_majority_voting = 0 #detecting right
        wrong_majority_voting = 0 #detecting wrong
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for index in xrange(0,int(0.1*len(data))):
            clf_result = int(clf_lgbm.predict([testX[index]])[0])
            clf_result += int(clf_svm.predict([testX[index]])[0])
            clf_result += int(clf_rf.predict([testX[index]])[0])
            clf_result += int(clf_ada.predict([testX[index]])[0])
            #clf_result += int(clf_mlp.predict([testX[index]])[0])
            clf_result += int(clf_knn.predict([testX[index]])[0])

            if clf_result >= 1:
                print '[1] This application is MALWARE'
                if testY[index] == 1:
                    right_majority_voting += 1
                    tp += 1
                else:
                    wrong_majority_voting += 1
                    fp += 1
            else:
                print '[0] This application is BENIGN'
                if testY[index] == 0:
                    right_majority_voting += 1
                    tn += 1
                else:
                    wrong_majority_voting += 1
                    fn += 1
            print '[%d] This application actually' %testY[index]
        print right_majority_voting,wrong_majority_voting
        print '[tp]:%d [fp]:%d [tn]:%d [fn]:%d' %(tp,fp,tn,fn)
        print 'Ensemble accuracy:{:.3f}%'.format(right_majority_voting*1.0/(right_majority_voting+wrong_majority_voting))
            
    else:
        print '[+] Usage: python {} <file_path>'.format(__file__)
