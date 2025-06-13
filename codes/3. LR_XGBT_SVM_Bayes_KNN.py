from sklearn.model_selection import KFold
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

filepath = 'C:\\project\\python\\pythonProject\\re\\keti\\dataset\\training data\\'
filelabel = 'HCGCN-qi-regulating herb pairs.content'


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

    # XGBoost训练并记录每10个epoch的结果
    for epoch in range(1, 501):
        bst = xgb.train(param, dtrain, num_boost_round=epoch)

        # 训练集预测和loss
        y_pred_train = (bst.predict(dtrain) >= 0.5).astype(int)
        y_pred_prob_train = bst.predict(dtrain)
        train_loss = metrics.log_loss(y_train, y_pred_prob_train)

        # 测试集预测和loss
        y_pred_test = (bst.predict(dtest) >= 0.5).astype(int)
        y_pred_prob_test = bst.predict(dtest)
        test_loss = metrics.log_loss(y_test, y_pred_prob_test)
        tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr = calculate_metrics(y_test, y_pred_test)

        if epoch % 10 == 0 or epoch == 500:
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'tpr': tpr,
                'fpr': fpr
            }
            write_epoch_results_to_file(epoch_data, 'XGBoost', fold)
            all_epoch_results['XGBoost'][epoch].append(epoch_data)

    # 最终评估
    bst = xgb.train(param, dtrain, num_boost_round=500)
    y_pred_test = (bst.predict(dtest) >= 0.5).astype(int)
    y_pred_prob_train = bst.predict(dtrain)
    train_loss = metrics.log_loss(y_train, y_pred_prob_train)
    y_pred_prob_test = bst.predict(dtest)
    test_loss = metrics.log_loss(y_test, y_pred_prob_test)
    tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr = calculate_metrics(y_test, y_pred_test)
    all_test_results['XGBoost'].append(
        (tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr, train_loss, test_loss))
    write_results_to_file({
        'fold': fold, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'auc': auc,
        'tpr': tpr, 'fpr': fpr, 'train_loss': train_loss, 'test_loss': test_loss
    }, 'XGBoost')

    # 处理其他模型
    for model_name, model in models.items():
        if model_name == 'XGBoost':
            continue

        print(f"\nTraining {model_name}...")

        # 对于可以记录训练过程的模型（如Logistic Regression）
        if model_name == 'Logistic Regression':
            # 自定义训练过程以记录epoch结果
            model = LogisticRegression(max_iter=1000, warm_start=True, verbose=0)
            n_epochs = 500  # 设置总epoch数

            for epoch in range(1, n_epochs + 1):
                if epoch == 1:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                if epoch % 10 == 0 or epoch == n_epochs:
                    # 训练集预测和loss
                    y_pred_train = model.predict(X_train)
                    y_pred_prob_train = model.predict_proba(X_train)[:, 1]
                    train_loss = metrics.log_loss(y_train, y_pred_prob_train)

                    # 测试集预测和loss
                    y_pred_test = model.predict(X_test)
                    y_pred_prob_test = model.predict_proba(X_test)[:, 1]
                    test_loss = metrics.log_loss(y_test, y_pred_prob_test)
                    tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr = calculate_metrics(y_test, y_pred_test)

                    epoch_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'tpr': tpr,
                        'fpr': fpr
                    }
                    write_epoch_results_to_file(epoch_data, model_name, fold)
                    all_epoch_results[model_name][epoch].append(epoch_data)
        else:
            # 对于不支持epoch记录的模型，我们只记录最终结果
            model.fit(X_train, y_train)

            # 训练集预测和loss
            try:
                y_pred_train = model.predict(X_train)
                y_pred_prob_train = model.predict_proba(X_train)[:, 1]
                train_loss = metrics.log_loss(y_train, y_pred_prob_train)
            except:
                train_loss = float('nan')

            # 测试集预测和loss
            if model_name == 'SVM':
                y_pred_prob_test = model.predict_proba(X_test)[:, 1]
                y_pred_test = (y_pred_prob_test >= 0.5).astype(int)
            else:
                y_pred_test = model.predict(X_test)

            try:
                y_pred_prob_test = model.predict_proba(X_test)[:, 1]
                test_loss = metrics.log_loss(y_test, y_pred_prob_test)
            except:
                test_loss = float('nan')

            tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr = calculate_metrics(y_test, y_pred_test)

            epoch_data = {
                'epoch': 500,  # 标记为最终epoch
                'train_loss': train_loss,
                'test_loss': test_loss,
                'tpr': tpr,
                'fpr': fpr
            }
            write_epoch_results_to_file(epoch_data, model_name, fold)
            all_epoch_results[model_name][500].append(epoch_data)

        # 计算并存储结果
        all_test_results[model_name].append(
            (tn, fp, fn, tp, acc, recall, precision, f1, auc, tpr, fpr, train_loss, test_loss))
        write_results_to_file({
            'fold': fold, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1, 'auc': auc,
            'tpr': tpr, 'fpr': fpr, 'train_loss': train_loss, 'test_loss': test_loss
        }, model_name)

# 计算并输出平均测试结果
average_results = {}
for model_name, results in all_test_results.items():
    # 计算各项指标的平均值
    avg_tn = np.mean([r[0] for r in results])
    avg_fp = np.mean([r[1] for r in results])
    avg_fn = np.mean([r[2] for r in results])
    avg_tp = np.mean([r[3] for r in results])
    avg_acc = np.mean([r[4] for r in results])
    avg_recall = np.mean([r[5] for r in results])
    avg_precision = np.mean([r[6] for r in results])
    avg_f1 = np.mean([r[7] for r in results])
    avg_auc = np.mean([r[8] for r in results])
    avg_tpr = np.mean([r[9] for r in results])
    avg_fpr = np.mean([r[10] for r in results])
    avg_train_loss = np.mean([r[11] for r in results if not np.isnan(r[11])])
    avg_test_loss = np.mean([r[12] for r in results if not np.isnan(r[12])])

    average_results[model_name] = {
        'avg_tn': avg_tn, 'avg_fp': avg_fp, 'avg_fn': avg_fn, 'avg_tp': avg_tp,
        'avg_acc': avg_acc, 'avg_recall': avg_recall, 'avg_precision': avg_precision,
        'avg_f1': avg_f1, 'avg_auc': avg_auc, 'avg_tpr': avg_tpr, 'avg_fpr': avg_fpr,
        'avg_train_loss': avg_train_loss, 'avg_test_loss': avg_test_loss
    }

# 计算并输出平均epoch结果
average_epoch_results = {model_name: {} for model_name in models}
for model_name, epochs_data in all_epoch_results.items():
    for epoch, fold_data in epochs_data.items():
        # 计算每个epoch的平均值
        avg_train_loss = np.mean([d['train_loss'] for d in fold_data if not np.isnan(d['train_loss'])])
        avg_test_loss = np.mean([d['test_loss'] for d in fold_data if not np.isnan(d['test_loss'])])
        avg_tpr = np.mean([d['tpr'] for d in fold_data])
        avg_fpr = np.mean([d['fpr'] for d in fold_data])

        average_epoch_results[model_name][epoch] = {
            'avg_train_loss': avg_train_loss,
            'avg_test_loss': avg_test_loss,
            'avg_tpr': avg_tpr,
            'avg_fpr': avg_fpr
        }

# 写入平均结果
write_averages_to_file(average_results)
write_average_epoch_results_to_file(average_epoch_results)

print("\n所有测试结果已保存到 model_test_results.csv 文件")
print("包含5折交叉验证的详细结果和平均结果")
print("\n训练过程中的TPR、FPR和loss值已保存到 model_epoch_results.csv 文件")
print("包含每10个epoch的结果和5折交叉验证的平均结果")