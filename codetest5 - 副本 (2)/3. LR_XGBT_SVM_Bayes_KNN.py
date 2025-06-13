from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")

filepath = 'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\'
filelabel = 'HCGCN-all herb pairs.content'


def Evaluating_Indicator(y_true, y_pred):
    '''
     TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    acc = (TP + TN)/(TN + FP + FN + TP)
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)
    F1 = 2/(1.0/precision + 1.0/recall)
    auc = roc_auc_score(y_true,y_pred)
    '''
    # 打印混淆矩阵
    print((confusion_matrix(y_true, y_pred).ravel()))

    # 计算准确率
    acc = metrics.accuracy_score(y_true, y_pred)
    # 计算召回率
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    # 计算精确率
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    # 计算F1值
    F1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    # 计算AUC值
    auc = metrics.roc_auc_score(y_true, y_pred)

    return acc, recall, precision, F1, auc


def write_result_to_file(acc, recall, precision, F1, auc):# 写入结果到文件
    with open('rs.csv', 'a') as fw:#写入指标
        fw.write(f"{acc},{recall},{precision},{F1},{auc}\n")#写入指标


df = pd.read_csv(filepath + filelabel, header=None, encoding='gbk', sep='\t')# 读取数据
df = df.replace('yes', 1)# 替换yes为1
df = df.replace('no', 0)# 替换no为0

x = df.iloc[:,
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
     32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]].values# 读取特征
y = df.iloc[:, 47].values# 读取标签

kf = KFold(n_splits=5, shuffle=False)# K折交叉验证
for train_index, test_index in kf.split(df):# 5折交叉验证
    X_train = x[train_index]# 读取训练集和测试集
    y_train = y[train_index]# 读取训练集和测试集
    X_test = x[test_index]
    y_test = y[test_index]

    # XGBoost
    dtrain = xgb.DMatrix(data=X_train, label=y_train)# 构建训练集和测试集
    dtest = xgb.DMatrix(data=X_test, label=y_test)# 构建训练
    param = {'max_depth': 6, 'eta': 0.3, 'verbosity': 1, 'lambda': 1, 'gamma': 1, 'subsample': 0.5,
             'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': 4, 'min_child_weight': 12, 'seed': 2}# 参数
    evallist = [(dtest, 'eval'), (dtrain, 'train')]# 构建验证集和训练集
    num_round = 85# 训练轮次
    bst = xgb.train(params=param, dtrain=dtrain, num_boost_round=num_round, evals=evallist)# 训练模型

    threshold = 0.5# 阈值
    y_pred_train = (bst.predict(dtrain) >= threshold) * 1# 预测训练集和测试集
    y_pred_test = (bst.predict(dtest) >= threshold) * 1

    print('XGBoost:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_train = log_reg.predict(X_train)
    y_pred_test = log_reg.predict(X_test)

    print('Logistic Regression:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)

    # Naive Bayes
    bayes_model = BernoulliNB()
    bayes_model.fit(X_train, y_train)
    y_pred_train = bayes_model.predict(X_train)
    y_pred_test = bayes_model.predict(X_test)

    print('Naive Bayes:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)

    print('KNN:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)

    clf = SVC(kernel='rbf', C=1, gamma=0.01, probability=True)
    svm = clf.fit(X_train, y_train)

    y_pred_prob_train = svm.predict_proba(X_train)[:, 1]
    y_pred_prob_test = svm.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_train = (y_pred_prob_train >= threshold).astype(int)
    y_pred_test = (y_pred_prob_test >= threshold).astype(int)

    print('SVM:')
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_train, y_pred_train)
    write_result_to_file(acc, recall, precision, F1, auc)
    acc, recall, precision, F1, auc = Evaluating_Indicator(y_test, y_pred_test)
    write_result_to_file(acc, recall, precision, F1, auc)
