# MobiSentry
add_db.py: Entry program, starts a task to extract features from apk files and stores them into db.

core2.py: focusing on extracting features.

create_data.py: focusing on obtainning features from db and create feature vectors for machine learning methods.

feature_selection.py: selecting important features to reduce the size of vectors, and to quicken the learning time.

grid_search_*.py: using gridsearch method for optimized parameters selection.

model_compare.py: comparing the detecting accuracy among the selected mechine learning methods.

plot_*.py: customized tool for ploting the AUC and ROC curves.

predict.py: This code classifies given apks into Malware/Benign using Malware-override rule for SVM ADABoosting RandormForest LightGBM.

predict_compare_majority_voting.py: This code classifies given apks into Malware/Benign using Majority Voting rule for SVM ADABoosting RandormForest LightGBM.

rbf_print.py: customized tool for logging.

We are happy to share our dataset. However, on the one hand, the whole dataset samples are really huge (184,486 benign applications and 21,306 malware samples, and nearly 2TB.), which makes it hard to share online. On the other hand, in order to prevent any misuse of malware samples, we kindly ask you to send us a mail to state your identity and research scope. We will then send you the dataset.
