from sklearn.model_selection import KFold
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

filepath = 'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\'
filelabel = 'HCGCN-all herb pairs.content'


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = metrics.accuracy_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    auc = metrics.roc_auc_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr


def write_roc_data_to_file(roc_data, model_name, fold):
    """写入ROC曲线数据到文件"""
    with open('roc_data_for_spss.csv', 'a') as f:
        if fold == 1 and model_name == 'XGBoost':  # 只在第一次写入表头
            f.write("Model,Fold,FPR,TPR\n")
        for fpr, tpr in zip(roc_data['fpr'], roc_data['tpr']):
            f.write(f"{model_name},{fold},{fpr:.6f},{tpr:.6f}\n")


def write_epoch_results_to_file(epoch_results, model_name, fold):
    """写入epoch结果到文件"""
    with open('model_epoch_results.csv', 'a') as f:
        if fold == 1 and epoch_results['epoch'] == 10:  # 只在第一次写入表头
            f.write("Model,Fold,Epoch,Train_Loss,Test_Loss,TPR,FPR\n")
        f.write(f"{model_name},{fold},{epoch_results['epoch']},{epoch_results['train_loss']:.4f},"
                f"{epoch_results['test_loss']:.4f},{epoch_results['tpr']:.4f},{epoch_results['fpr']:.4f}\n")


def write_average_epoch_results_to_file(average_results):
    """写入平均epoch结果到文件"""
    with open('model_epoch_results.csv', 'a') as f:
        f.write("\nAverage Results Across 5 Folds:\n")
        f.write("Model,Epoch,Avg_Train_Loss,Avg_Test_Loss,Avg_TPR,Avg_FPR\n")
        for model_name, epochs_data in average_results.items():
            for epoch, metrics in epochs_data.items():
                f.write(f"{model_name},{epoch},{metrics['avg_train_loss']:.4f},{metrics['avg_test_loss']:.4f},"
                        f"{metrics['avg_tpr']:.4f},{metrics['avg_fpr']:.4f}\n")


def write_results_to_file(results, model_name):
    """写入结果到文件"""
    with open('model_test_results.csv', 'a') as f:
        if results['fold'] == 1:  # 只在第一次写入表头
            f.write("Model,Fold,TN,FP,FN,TP,Accuracy,Recall,Precision,F1,AUC,TPR,FPR,Train_Loss,Test_Loss\n")
        f.write(f"{model_name},{results['fold']},{results['tn']},{results['fp']},{results['fn']},{results['tp']},"
                f"{results['acc']:.4f},{results['recall']:.4f},{results['precision']:.4f},{results['f1']:.4f},"
                f"{results['auc']:.4f},{results['tpr']:.4f},{results['fpr']:.4f},"
                f"{results['train_loss']:.4f},{results['test_loss']:.4f}\n")


def write_averages_to_file(averages):
    """写入平均结果到文件"""
    with open('model_test_results.csv', 'a') as f:
        f.write("\nAverage Results Across 5 Folds:\n")
        f.write(
            "Model,Avg_TN,Avg_FP,Avg_FN,Avg_TP,Avg_Accuracy,Avg_Recall,Avg_Precision,Avg_F1,Avg_AUC,Avg_TPR,Avg_FPR,Avg_Train_Loss,Avg_Test_Loss\n")
        for model_name, metrics in averages.items():
            f.write(
                f"{model_name},{metrics['avg_tn']:.1f},{metrics['avg_fp']:.1f},{metrics['avg_fn']:.1f},{metrics['avg_tp']:.1f},"
                f"{metrics['avg_acc']:.4f},{metrics['avg_recall']:.4f},{metrics['avg_precision']:.4f},{metrics['avg_f1']:.4f},"
                f"{metrics['avg_auc']:.4f},{metrics['avg_tpr']:.4f},{metrics['avg_fpr']:.4f},"
                f"{metrics['avg_train_loss']:.4f},{metrics['avg_test_loss']:.4f}\n")


# 读取数据
df = pd.read_csv(filepath + filelabel, header=None, encoding='gbk', sep='\t')
df = df.replace('yes', 1)
df = df.replace('no', 0)

x = df.iloc[:, 1:47].values  # 特征列
y = df.iloc[:, 47].values  # 标签列

# 初始化模型
models = {
    'XGBoost': None,
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': BernoulliNB(),
    'KNN': KNeighborsClassifier(n_neighbors=9),
    'SVM': SVC(kernel='rbf', C=1, gamma=0.01, probability=True)
}

# 存储每个模型的测试结果
all_test_results = {model_name: [] for model_name in models}
# 存储每个模型的epoch结果
all_epoch_results = {model_name: defaultdict(list) for model_name in models}
# 存储ROC曲线数据
all_roc_data = {model_name: [] for model_name in models}

# 清空或创建ROC数据文件
with open('roc_data_for_spss.csv', 'w') as f:
    f.write("")  # 清空文件

kf = KFold(n_splits=5, shuffle=False)
for fold, (train_index, test_index) in enumerate(kf.split(df), 1):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"\n=== Fold {fold} ===")

    # XGBoost特殊处理
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 6, 'eta': 0.3, 'verbosity': 1, 'lambda': 1, 'gamma': 1,
             'subsample': 0.5, 'colsample_bytree': 0.8, 'objective': 'binary:logistic',
             'nthread': 4, 'min_child_weight': 12, 'seed': 2}

    # XGBoost训练
    bst = xgb.train(param, dtrain, num_boost_round=200)

    # 获取预测概率
    y_pred_prob_test = bst.predict(dtest)

    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
    roc_data = {'fpr': fpr, 'tpr': tpr}
    write_roc_data_to_file(roc_data, 'XGBoost', fold)
    all_roc_data['XGBoost'].append(roc_data)

    # 处理其他模型
    for model_name, model in models.items():
        if model_name == 'XGBoost':
            continue

        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)

        # 获取预测概率
        if model_name == 'SVM':
            y_pred_prob_test = model.predict_proba(X_test)[:, 1]
        else:
            try:
                y_pred_prob_test = model.predict_proba(X_test)[:, 1]
            except:
                y_pred_prob_test = model.predict(X_test)  # 对于没有概率预测的模型

        # 计算ROC曲线数据
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
        roc_data = {'fpr': fpr, 'tpr': tpr}
        write_roc_data_to_file(roc_data, model_name, fold)
        all_roc_data[model_name].append(roc_data)

# 计算平均ROC曲线数据并写入文件
with open('roc_data_for_spss_avg.csv', 'w') as f:
    f.write("Model,FPR,TPR\n")  # 表头

    for model_name, roc_list in all_roc_data.items():
        # 收集所有FPR点
        all_fpr = np.unique(np.concatenate([roc['fpr'] for roc in roc_list]))

        # 对每个FPR点计算平均TPR
        mean_tpr = np.zeros_like(all_fpr)
        for roc in roc_list:
            mean_tpr += np.interp(all_fpr, roc['fpr'], roc['tpr'])
        mean_tpr /= len(roc_list)

        # 写入平均ROC数据
        for fpr, tpr in zip(all_fpr, mean_tpr):
            f.write(f"{model_name},{fpr:.6f},{tpr:.6f}\n")

print("\n所有测试结果已保存到 model_test_results.csv 文件")
print("训练过程中的TPR、FPR和loss值已保存到 model_epoch_results.csv 文件")
print("\nROC曲线数据已保存到 roc_data_for_spss.csv (每折数据) 和 roc_data_for_spss_avg.csv (平均数据)")
print("您可以在SPSS中导入这些文件绘制ROC曲线:")
print("1. 使用 roc_data_for_spss.csv 绘制各折ROC曲线")
print("2. 使用 roc_data_for_spss_avg.csv 绘制平均ROC曲线")