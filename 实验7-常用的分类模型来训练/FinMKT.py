import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# some usable model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings('ignore')
# os.chdir('/lqq/FinMKT/venv')


def data_preprocess(data):
    # your code here
    # print("data.shape =", data.shape)
    # print(data.head())
    # print(data.info())

    x = pd.get_dummies(data)
    # print("x.shape =", x.shape)
    # print(x.head())
    # print(x.info())
    # 1：缺失值处理
    # 不处理／丢弃／填充
    # age-average,sex/marriage/study etc-0
    # 没有缺失值，所以跳过这一步

    # 2：字符串数据处理:建立字符串索引
    # one-hot编码、建立字符串索引（转换为出现频率）

    # 3：特征二值化:通过设置阈值，把数值sum_ckcs的特征转换为布尔值即客户是否会订购存款。
    #             将sum_ckcs大于0的值设为1表示该 客户会订购，sum_ckcs等于0的值设为0表示该客户不会订购存款。

    # 4：数据归一化：min-max将所有数据缩放到0-1之间


    # 通过删除均值和缩放到单位方差来标准化特征
    scaler = StandardScaler()
    x = scaler.fit_transform(x)



    # your code end
    return x

def predictKN(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'

    var = VarianceThreshold(threshold=1)
    x_train = var.fit_transform(x_train)
    x_test = var.transform(x_test)

    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # your code here end

    return y_pred

def predictSVM(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'

    clf = SVC()
    clf.fit(x_train, y_train)

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

    y_pred = clf.predict(x_test)

    # your code here end

    return y_pred

def predictLR(x_train, x_test, y_train):

    # your code here begin
    # train your model on 'x_train' and 'x_test'
    # predict on 'y_train' and get 'y_pred'

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # your code here end

    return y_pred


def split_data(data):
    y = data.y
    x = data.loc[:, data.columns != 'y']
    x = data_preprocess(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test

def print_result(y_test, y_pred):
    report = confusion_matrix(y_test, y_pred)
    precision = report[1][1] / (report[:, 1].sum())
    recall = report[1][1] / (report[1].sum())
    print('model precision:' + str(precision)[:4] + ' recall:' + str(recall)[:4])

    F1_score = 2*precision*recall / (precision+recall)
    print('F1_score:' + str(F1_score)[:4])


    # plt.figure()
    # plt.plot(np.arange(len(y_pred)), y_test, 'go-', label='true value')
    # plt.plot(np.arange(len(y_pred)), y_pred, 'ro-', label='predict value')
    # plt.title('F1_score: %f' % F1_score)
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictKN(x_train, x_test, y_train)
    print("\n===KN===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictSVM(x_train, x_test, y_train)
    print("\n===SVM===")
    print_result(y_test, y_pred)

    data = pd.read_csv('bank-additional-full.csv', sep=';')
    x_train, x_test, y_train, y_test = split_data(data)
    y_pred = predictLR(x_train, x_test, y_train)
    print("\n===LR===")
    print_result(y_test, y_pred)



