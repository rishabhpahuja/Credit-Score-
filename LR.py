import numpy as np
import pandas as pd
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def dbscan(new_X_data,y_label):
    i=1
    while True:
        DBSCAN_model=DBSCAN(eps=i, min_samples=80).fit(new_X_data)
        labels=DBSCAN_model.labels_
        i+=1
        print(len(np.where(labels==-1)[0]))
        if len(np.where(labels==-1)[0])<=0.06*len(new_X_data) and len(np.where(labels==-1)[0])>=0.01*len(new_X_data):
            print(len(np.where(labels==-1)[0]))
            print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',i)
            break
    X_datatset_no_outlier=new_X_data.loc[labels!=-1,:]
    y_label_no_outlier=y_label.loc[labels!=-1]
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print((X_data))
    print((X_datatset_no_outlier))
    plot_outliers(new_X_data,labels)
    return X_datatset_no_outlier, y_label_no_outlier

def pca(X_data):
    cov=(X_data.T@X_data)/(len(X_data)-1)
    sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

    variance=np.zeros(len(X_data.columns))
    for i in range(len(X_data.columns)):
        variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)

    print(variance)

    new_X_data=X_data@sorted_eig_vecs[:,:2]
    return new_X_data    

def LR(X_train,X_test,y_train,y_test,a):
    scores=list()
    for k in range(1,35):
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test,y_pred))
    plot_score(scores,a)

def plot_score(scores,a):
    #plt.subplot(1, 2, 1)
    plt.plot(range(1,35),scores)
    plt.title(a)
    plt.show()
    
def plot_outliers(new_X_data,labels):
    plt.subplot(1, 2, 1)
    outliers=np.where(labels==-1)
    plt.title('Outliers Detected')
    plt.plot(new_X_data.loc[outliers,0],new_X_data.loc[outliers,1],'x')

    plt.subplot(1, 2, 2)
    plt.title('Original dataset after PCA')
    plt.plot(new_X_data.loc[:,0],new_X_data.loc[:,1],'x')
    plt.show()

def main(X_data,y_label):
    # cov=(X_data.T@X_data)/(len(X_data)-1)
    # sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

    # variance=np.zeros(len(X_data.columns))
    # for i in range(len(X_data.columns)):
    #     variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)

    # print(variance)

    # new_X_data=X_data@sorted_eig_vecs[:,:2]
    new_X_data = pca(X_data)
    '''
    Performing KNN
    '''

    #scores=list()
    X_train,X_test,y_train,y_test=train_test_split(new_X_data,y_label,test_size=0.2,random_state=42)
    a = 'With outlier'
    LR(X_train,X_test,y_train,y_test,a)
    # for k in range(1,35):
    #     knn=KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train,y_train)
    #     y_pred=knn.predict(X_test)
    #     scores.append(metrics.accuracy_score(y_test,y_pred))

    # plt.subplot(1, 2, 1)
    # plt.title('With Outlier')
    # plt.plot(range(1,35),scores)



    '''
    KNN after outlier detection
    '''

    '''
    Outlier Detection using DBSCAN
    '''
    # i=1
    # while True:
    #     DBSCAN_model=DBSCAN(eps=i, min_samples=80).fit(new_X_data)
    #     labels=DBSCAN_model.labels_
    #     i+=1
    #     print(len(np.where(labels==-1)[0]))
    #     if len(np.where(labels==-1)[0])<=0.06*len(new_X_data) and len(np.where(labels==-1)[0])>=0.01*len(new_X_data):
    #         print(len(np.where(labels==-1)[0]))
    #         print('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',i)
    #         break
    # X_datatset_no_outlier=new_X_data.loc[labels!=-1,:]
    # y_label_no_outlier=y_label.loc[labels!=-1]
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # # print((X_data))
    # print((X_datatset_no_outlier))
    X_datatset_no_outlier, y_label_no_outlier = dbscan(new_X_data,y_label)

    '''
    PCA on datatset without outlier
    '''

    # cov=(X_datatset_no_outlier.T@X_datatset_no_outlier)/(len(X_datatset_no_outlier)-1)
    # sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

    # variance=np.zeros(len(X_datatset_no_outlier.columns))
    # for i in range(len(X_datatset_no_outlier.columns)):
    #     variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)

    # new_australian_dataset_no_outlier=X_datatset_no_outlier@sorted_eig_vecs[:,:2]


    X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(X_datatset_no_outlier,y_label_no_outlier,test_size=0.2,random_state=42)
    b = 'Without outlier'
    LR(X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier,b)
    # scores_no_outlier=list()
    # for k in range(1,35):
    #     knn=KNeighborsClassifier(n_neighbors=k,leaf_size=2,p=2)
    #     knn.fit(X_train_no_outlier,y_train_no_outlier)
    #     y_pred_no_outlier=knn.predict(X_test_no_outlier)
    #     scores_no_outlier.append(metrics.accuracy_score(y_test_no_outlier,y_pred_no_outlier))

    # plt.subplot(1, 2, 2)
    # plt.title('Without Outlier')
    # plt.plot(range(1,35),scores_no_outlier)
    # plt.show()

    # plt.subplot(1, 2, 1)
    # a=np.where(labels==-1)
    # plt.plot(new_X_data.loc[a,0],new_X_data.loc[a,1],'x')

    # plt.subplot(1, 2, 2)
    # plt.plot(new_X_data.loc[:,0],new_X_data.loc[:,1],'x')
    # plt.show()


if __name__ == '__main__':
    australian_dataset=pd.read_csv('australian.dat',header=None,sep=' ')
    print(australian_dataset)
    y_label=australian_dataset.loc[:,len(australian_dataset.columns)-1]
    X_data=australian_dataset.drop(australian_dataset.columns[[len(australian_dataset.columns)-1]],axis=1)
    main(X_data,y_label)
