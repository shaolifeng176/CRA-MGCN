# HCGCN
#code instructions HCGCN algorithm is based on GCN model ( https://github.com/tkipf/gcn) and ETCM data ) to predict the combination of traditional Chinese medicine. The specific use method is as follows :

##1 generate adjacency matrix.py generates the adjacency matrix according to the known drug pairs. The adjacency matrix is established according to the input fixed drug pair file txt.

##2 train.py generates training set and test set results based on the generated training set and test set results The confusion matrix and the values of acc, recall, precision, F1, auc are recorded respectively.

##3 LR _ XGBT _ SVM _ Bayes _ KNN.py generates training set and test set results of LR, XGBT and other models.

##4 predict.py to predict based on the trained model.th Probabilistic output and judgment output of unknown drug pairs are carried out.
