import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
#from skopt import BayesSearchCV
from nn import *
from Data_read import *
import seaborn as sns
from csv import writer

global acc_with_outliers, acc_without_outliers
acc_with_outliers = np.zeros((7,1))
acc_without_outliers = np.zeros((7,1))

# def plot_score(scores,a):
#     plt.plot(range(1,35),scores)
#     plt.title(a)
#     plt.show()
# def plot_hist():
#     global acc_with_outliers,acc_without_outliers
#     print(acc_without_outliers)
#     print(acc_with_outliers)
#     plt.figure("Accuracy Comparison Chart")
#     plt.hist([acc_with_outliers*100, acc_without_outliers*100], color=['r','b'], alpha=0.5, label=['With outliers','Without outliers'], x=['KNN', 'Logistic Regression', 'Adaboost', 'Random Forest', 'GNB', 'SVM', 'NN'])
    #sns.distplot(acc_with_outliers, label='With outliers', color="0.25")
    #sns.distplot(acc_without_outliers, label='Without outliers', color="0.25")
    # plt.legend()
    # plt.show()
    # df = np.concatenate((acc_with_outliers,acc_without_outliers),axis=1)
    # fig, ax = plt.subplots()
    # sns.histplot(
    # data=df, x='value', hue='name', multiple='dodge',
    # bins=range(1, 110, 10), ax=ax
#)
#ax.set_xlim([0, 100])

def write_text_file(line):
    # with open('accuracy.txt', 'a') as f:
    #     f.write(line)
    #     f.write('\n')
    x = line.split(",")
    with open('accuracy.csv', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(x)  
        f_object.close()


def plot_outliers(new_X_data,labels):
    # plt.subplot(1, 2, 1)
    # outliers=np.where(labels==-1)
    # plt.title('Outliers Detected')
    # plt.plot(new_X_data.loc[outliers,0],new_X_data.loc[outliers,1],'x')

    # plt.subplot(1, 2, 2)
    # plt.title('Original dataset after PCA')
    # plt.plot(new_X_data.loc[:,0],new_X_data.loc[:,1],'x')
    # plt.show()
    # new_x_data_outlier=new_X_data[np.where(labels==-1),:]
    print(len(labels))
    labels=(np.where(labels!=-1),1,labels)
    print(len(labels))
    sns.scatterplot(x=new_X_data.iloc[:,0], y=new_X_data.iloc[:,1], hue=labels, style=labels,palette='Set2')


def outlier_removal(new_X_data,y_label,min_outliers,max_outliers, Samples):
    i=1
    while True:
        DBSCAN_model=DBSCAN(eps=i, min_samples=Samples).fit(new_X_data)
        labels=DBSCAN_model.labels_
        i+=5
        print(len(np.where(labels==-1)[0]))
        if len(np.where(labels==-1)[0])<=max_outliers*len(new_X_data) and len(np.where(labels==-1)[0])>=min_outliers*len(new_X_data):
            print('min dist dbscan:',i)
            break
    X_datatset_no_outlier=new_X_data.loc[labels!=-1,:]
    y_label_no_outlier=y_label.loc[labels!=-1]
    plot_outliers(new_X_data,labels)
    return X_datatset_no_outlier, y_label_no_outlier

def pca(X_data):
    pca = PCA(n_components=2)
    new_X_data = pca.fit_transform(X_data)
    print(pca.explained_variance_ratio_)
    return pd.DataFrame(new_X_data)

def k_nearest(X_train,X_test,y_train,y_test,a,data_name):
    scores=list()
    for k in range(1,35):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,y_pred))

    # plt.plot(range(1,35),scores)
    plt.title(a)
    plt.show()
    string = 'KNN,' + str(scores[-1]) + ',' + data_name
    write_text_file(string)
    return scores[-1]
    #plot_score(scores,a)

def Logistic_Reg(X_train,X_test,y_train,y_test,data_name):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Logistic Regression Accuracy = ', metrics.accuracy_score(y_test,y_pred))
    string = 'Logistic Regression,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name
    write_text_file(string)
    return metrics.accuracy_score(y_test,y_pred)
    # params = dict()
    # params['C'] = (1e-6, 100.0, 'log-uniform')
    # # define evaluation
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # define the search
    # search = BayesSearchCV(estimator=LogisticRegression(), search_spaces=params, n_jobs=-1, cv=cv)
    # # perform the search
    # search.fit(X_train, y_train)
    # # report the best result
    # print(search.best_score_)


def adaboost_classifier(X_train,X_test,y_train,y_test,data_name):
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Adaboost,' +  str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name
    write_text_file(string)
    print('Adaboost classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def random_forest(X_train,X_test,y_train,y_test,data_name):
    clf = RandomForestClassifier(max_depth=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Random Forest,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name
    write_text_file(string)
    print('Random Forest classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def gaussian_nb(X_train,X_test,y_train,y_test,data_name):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Gaussian Naive Bayes,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name
    write_text_file(string)
    print('Gaussian Naive Bayes classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def svm(X_train,X_test,y_train,y_test,data_name):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Support Vector Machine,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name
    write_text_file(string)
    print('Support Vector Machine classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

    # params = dict()
    # params['C'] = (1e-6, 100.0, 'log-uniform')
    # params['gamma'] = (1e-6, 100.0, 'log-uniform')
    # params['degree'] = (1,5)
    # params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
    # # define evaluation
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # define the search
    # search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv)
    # # perform the search
    # search.fit(X_train, y_train)
    # # report the best result
    # print(search.best_score_)

def main(X_data,y_label,min_outliers,max_outliers, Samples, data_name):
    new_X_data = pca(X_data)

    #Performing KNN
    X_train,X_test,y_train,y_test=train_test_split(new_X_data,y_label,test_size=0.2,random_state=42)
    tit = 'With outlier'
    acc_with_outliers[0] = k_nearest(X_train,X_test,y_train,y_test,tit,data_name)
    acc_with_outliers[1] = Logistic_Reg(X_train,X_test,y_train,y_test,data_name)
    acc_with_outliers[2] = adaboost_classifier(X_train,X_test,y_train,y_test,data_name)
    acc_with_outliers[3] = random_forest(X_train,X_test,y_train,y_test,data_name)
    acc_with_outliers[4] = gaussian_nb(X_train,X_test,y_train,y_test,data_name)
    acc_with_outliers[5] = svm(X_train,X_test,y_train,y_test,data_name)
    acc_with_outliers[6] = nn(X_train,X_test,y_train,y_test,data_name)
    '''
    KNN after outlier detection
    '''

    '''
    Outlier Detection using DBSCAN
    '''
    X_datatset_no_outlier, y_label_no_outlier = outlier_removal(new_X_data,y_label,min_outliers,max_outliers, Samples)


    X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(X_datatset_no_outlier,y_label_no_outlier,test_size=0.2,random_state=42)
    tit = 'Without outlier'
    acc_without_outliers[0] = k_nearest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,tit,data_name)
    acc_without_outliers[1] = Logistic_Reg(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)
    acc_without_outliers[2] = adaboost_classifier(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)
    acc_without_outliers[3] = random_forest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)
    acc_without_outliers[4] = gaussian_nb(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)
    acc_without_outliers[5] = svm(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)
    acc_without_outliers[6] = nn(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name)

if __name__ == '__main__':
    data_name = input("Enter Dataset Name:")
    X_data, y_label, min_outliers, max_outliers, Samples = eval(data_name + "()")
    main(X_data,y_label,min_outliers,max_outliers, Samples, data_name)
    # plot_hist()

