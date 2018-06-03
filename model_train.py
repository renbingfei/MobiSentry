'''
Author: Ashish Katlam
Description: This code trains the malware detection model

'''

from core2 import create_vector_single
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

def load_data():
	feature_vector = []
	# Data format
	# Column 0: Class label (1: Malware, 0: Benign)
	# Column 1-19: Features
	with open('final_data.csv','r') as fp:
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

	# Check for arguments
	if len(sys.argv) > 1:
		evaluation_metrics = sys.argv[1]

		# Load the data
		data = load_data()

		# Shuffle the data
		random.shuffle(data)

		# If evaluation metrics is using training and testing
		if evaluation_metrics == "train_test":
			# Divide the data into training and testing in 60:40
			trainLength = int(0.88888888*len(data))
			# Training Data
			trainX = [x[:-1] for x in data[:trainLength]]
			trainY = [y[-1] for y in data[:trainLength]]

			# Testing Data
			testX = [x[:-1] for x in data[trainLength:]]
			testY = [y[-1] for y in data[trainLength:]]


			# Perform training
			print 'Training the data....'
			train_result = {'timestamp': datetime.datetime.now(),'alg_type': 'svm'}
			clf = svm.SVC()
			clf.set_params(kernel='rbf').fit(trainX, trainY)

			print 'default Accuracy: {:.3f}%'.format(clf.score(testX, testY)*100)
                        predicted = clf.predict(testX)
                        print metrics.classification_report(testY,predicted)
                        print metrics.confusion_matrix(testY,predicted)
	
                        # test class_weight
                        clf = svm.SVC(class_weight={1:100,0:11.622})
                        clf.set_params(kernel='rbf').fit(trainX,trainY)
                        print 'ratio Accuracy:{:.3f}%'.format(clf.score(testX,testY)*100)
                        predicted = clf.predict(testX)
                        print metrics.classification_report(testY,predicted)
                        print metrics.confusion_matrix(testY,predicted)
	

                        clf = svm.SVC(class_weight='balanced')
                        clf.set_params(kernel='rbf').fit(trainX,trainY)
                        print 'balanced Accuracy:{:.3f}%'.format(clf.score(testX,testY)*100)

                        # classification report
                        predicted = clf.predict(testX)
                        print metrics.classification_report(testY,predicted)
                        print metrics.confusion_matrix(testY,predicted)
			# Save the trained model so that it can be used later
			pickle.dump(clf, open( "new_train_data.p", "wb" ))
			print 'Model trained and saved.'


		else:
			X = [x[:-1] for x in data]
			Y = [y[-1] for y in data]
			k_fold = KFold(5)
			clf = svm.SVC(probability=True)
                        #clf = svm.SVC()
			results = []
			i = 1
			
			print 'Performing 5-fold cross validation....'
			
			for train, test in k_fold.split(X):
				x = [X[ind] for ind in train]
				y = [Y[ind] for ind in train]
				x_test = [X[ind] for ind in test]
				y_test = [Y[ind] for ind in test]

				clf.set_params(kernel='rbf').fit(x,y)
				score = clf.score(x_test,y_test)
				results.append(score)
				print "[fold {0}]  score: {1:.5f}".format(i, score)
				i+=1
                        

			print 'Mean Score: {}'.format(sum(results)/len(results))

			# Dump the model	
			pickle.dump(clf, open("kfold_train_data.p", "wb"))
			print "Model trained and saved."
                        if len(sys.argv)>2 and sys.argv[2]=='roc':
                           # print roc_curve and auc score
                           mean_tpr = 0.0
                           mean_fpr = np.linspace(0,1,100)
                           all_tpr = []
                           index = 1
                           for train, test in k_fold.split(X):
           			x = [X[ind] for ind in train]
           			y = [Y[ind] for ind in train]
           			x_test = [X[ind] for ind in test]
           			y_test = [Y[ind] for ind in test]

           		        probas_ = clf.set_params(kernel='rbf').fit(x,y).predict_proba(x_test)
                                fpr,tpr,thresholds = roc_curve(y_test,probas_[:,1])
                                mean_tpr += interp(mean_fpr,fpr,tpr)
                                mean_tpr[0] = 0.0
                                roc_auc = auc(fpr,tpr)
                                plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)' %(index,roc_auc))
                                index += 1
                           plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label='Luck')
                           print 'begin to plot ROC figure'

                           mean_tpr /= 5
                           mean_tpr[-1] = 1.0
                           mean_auc = auc(mean_fpr,mean_tpr)

                           plt.plot(mean_fpr,mean_tpr,'k--',label='Mean ROC (area = %0.2f)'% mean_auc,lw=2)
                           plt.xlim([-0.05,1.05])
                           plt.ylim([-0.05,1.05])
                           plt.xlabel('False Positive Rate')
                           plt.ylabel('True Positive Rate')
                           plt.title('Receiver operating characteristic')
                           plt.legend(loc='lower right')
                           #plt.show()
                           savefig('roc')
    


	else:
		print '[+] Usage: python {} <train_test/k-fold[roc]>'.format(__file__)
	 

