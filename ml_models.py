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

# def plot_score(scores,a):
#     plt.plot(range(1,35),scores)
#     plt.title(a)
#     plt.show()
    
def plot_outliers(new_X_data,labels):
    plt.subplot(1, 2, 1)
    outliers=np.where(labels==-1)
    plt.title('Outliers Detected')
    plt.plot(new_X_data.loc[outliers,0],new_X_data.loc[outliers,1],'x')

    plt.subplot(1, 2, 2)
    plt.title('Original dataset after PCA')
    plt.plot(new_X_data.loc[:,0],new_X_data.loc[:,1],'x')
    plt.show()

def outlier_removal(new_X_data,y_label,min_outliers,max_outliers):
    i=1
    while True:
        DBSCAN_model=DBSCAN(eps=i, min_samples=80).fit(new_X_data)
        labels=DBSCAN_model.labels_
        i+=1
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

def k_nearest(X_train,X_test,y_train,y_test,a):
    scores=list()
    for k in range(1,35):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,y_pred))

    plt.plot(range(1,35),scores)
    plt.title(a)
    plt.show()
    #plot_score(scores,a)

def Logistic_Reg(X_train,X_test,y_train,y_test):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Logistic Regression Accuracy = ', metrics.accuracy_score(y_test,y_pred))

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


def adaboost_classifier(X_train,X_test,y_train,y_test):
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Adaboost classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))

def random_forest(X_train,X_test,y_train,y_test):
    clf = RandomForestClassifier(max_depth=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Random Forest classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))

def gaussian_nb(X_train,X_test,y_train,y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Gaussian Naive Bayes\' classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))

def svm(X_train,X_test,y_train,y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Support Vector Machine classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))

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

def main(X_data,y_label,min_outliers,max_outliers):
    new_X_data = pca(X_data)

    #Performing KNN
    X_train,X_test,y_train,y_test=train_test_split(new_X_data,y_label,test_size=0.2,random_state=42)
    tit = 'With outlier'
    k_nearest(X_train,X_test,y_train,y_test,tit)
    Logistic_Reg(X_train,X_test,y_train,y_test)
    adaboost_classifier(X_train,X_test,y_train,y_test)
    random_forest(X_train,X_test,y_train,y_test)
    gaussian_nb(X_train,X_test,y_train,y_test)
    svm(X_train,X_test,y_train,y_test)
    nn(X_train,X_test,y_train,y_test)
    '''
    KNN after outlier detection
    '''

    '''
    Outlier Detection using DBSCAN
    '''
    X_datatset_no_outlier, y_label_no_outlier = outlier_removal(new_X_data,y_label,min_outliers,max_outliers)


    X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(X_datatset_no_outlier,y_label_no_outlier,test_size=0.2,random_state=42)
    tit = 'Without outlier'
    k_nearest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,tit)
    Logistic_Reg(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)
    adaboost_classifier(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)
    random_forest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)
    gaussian_nb(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)
    svm(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)
    nn(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier)

if __name__ == '__main__':
    # australian_dataset=pd.read_csv('./Datasets/uci-australian.dat',header=None,sep=' ')
    # print(australian_dataset)
    # y_label=australian_dataset.loc[:,len(australian_dataset.columns)-1]
    # X_data=australian_dataset.drop(australian_dataset.columns[[len(australian_dataset.columns)-1]],axis=1)
    X_data, y_label, min_outliers, max_outliers = aus()
    main(X_data,y_label,min_outliers,max_outliers)


