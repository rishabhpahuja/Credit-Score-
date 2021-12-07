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

def write_text_file(line):
    x = line.split(",")
    with open('accuracy.csv', 'a', newline='') as f_object:  
        writer_object = writer(f_object)
        writer_object.writerow(x)  
        f_object.close()


def plot_outliers(new_X_data,labels):

    labels=np.where(labels!=-1,'Inlier','Outlier')
    print(len(new_X_data.iloc[:,0]))
    ax = sns.scatterplot(x=new_X_data.iloc[:,0], y=new_X_data.iloc[:,1], hue=labels, style=labels,palette='Set2',edgecolor=None)
    ax.set(xlabel="Feature 1",ylabel="Feature 2",title="Outlier Detection")
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.show()

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

def k_nearest(X_train,X_test,y_train,y_test,data_name,flag):
    scores=list()
    for k in range(1,35):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,y_pred))

    string = 'KNN,' + str(scores[-1]) + ',' + data_name+','+flag
    write_text_file(string)
    return scores[-1]

def Logistic_Reg(X_train,X_test,y_train,y_test,data_name,flag):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Logistic Regression Accuracy = ', metrics.accuracy_score(y_test,y_pred))
    string = 'Logistic Regression,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name+','+flag
    write_text_file(string)
    return metrics.accuracy_score(y_test,y_pred)

def adaboost_classifier(X_train,X_test,y_train,y_test,data_name,flag):
    clf = AdaBoostClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Adaboost,' +  str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name+','+flag
    write_text_file(string)
    print('Adaboost classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def random_forest(X_train,X_test,y_train,y_test,data_name,flag):
    clf = RandomForestClassifier(max_depth=150, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Random Forest,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name+','+flag
    write_text_file(string)
    print('Random Forest classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def gaussian_nb(X_train,X_test,y_train,y_test,data_name,flag):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Gaussian Naive Bayes,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name+','+flag
    write_text_file(string)
    print('Gaussian Naive Bayes classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def svm(X_train,X_test,y_train,y_test,data_name,flag):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    string = 'Support Vector Machine,' + str(metrics.accuracy_score(y_test,y_pred)) + ',' + data_name+','+flag
    write_text_file(string)
    print('Support Vector Machine classifier accuracy: ', metrics.accuracy_score(y_test,y_pred))
    return metrics.accuracy_score(y_test,y_pred)

def main(X_data,y_label,min_outliers,max_outliers, Samples, data_name):
    new_X_data = pca(X_data)

    X_train,X_test,y_train,y_test=train_test_split(new_X_data,y_label,test_size=0.2,random_state=42)
    
    flag = 'With Outlier'
    
    acc_with_outliers[0] = k_nearest(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[1] = Logistic_Reg(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[2] = adaboost_classifier(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[3] = random_forest(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[4] = gaussian_nb(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[5] = svm(X_train,X_test,y_train,y_test,data_name,flag)
    acc_with_outliers[6] = nn(X_train,X_test,y_train,y_test,data_name,flag)

    X_datatset_no_outlier, y_label_no_outlier = outlier_removal(new_X_data,y_label,min_outliers,max_outliers, Samples)

    X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(X_datatset_no_outlier,y_label_no_outlier,test_size=0.2,random_state=42)

    flag = 'Without Outlier'
    
    acc_without_outliers[0] = k_nearest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[1] = Logistic_Reg(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[2] = adaboost_classifier(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[3] = random_forest(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[4] = gaussian_nb(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[5] = svm(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)
    acc_without_outliers[6] = nn(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,data_name,flag)

if __name__ == '__main__':
    data_name = input("Enter Dataset Name:")
    X_data, y_label, min_outliers, max_outliers, Samples = eval(data_name + "()")
    main(X_data,y_label,min_outliers,max_outliers, Samples, data_name)


