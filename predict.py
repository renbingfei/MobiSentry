'''
Author: Ashish Katlam
Description: This code classifies a given apk into Malware/Benign
'''
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import pickle
import sys
from core2 import extract_features, create_vector_single

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
        # Extract the features for a given application
        features = extract_features(file_path)
        # Form the feature vector
        feature_vector = create_vector_single(features)
        # Load the pre configured feature model from a pickle file
        model = pickle.load(open("feature_model.p", "rb"))
        # Reduce the feature vector into size 12
        feature_vector_new = model.transform(feature_vector)


        # Load the pre-trained model from a pickle file
        clf = pickle.load( open( "kfold_train_data.p", "rb" ) )

        # Perform prediction using the model
        result = clf.predict(feature_vector_new)
        print 'prob:',clf.predict_proba(feature_vector_new)

        #print result
        
        if int(result[0]) == 1:
            print 'This application is MALWARE'
        else:
            print "This application is BENIGN"
    else:
        print '[+] Usage: python {} <file_path>'.format(__file__)
