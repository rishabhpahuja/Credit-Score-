import csv
import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import append
import pandas as pd
from csv import writer

from pandas.io.pytables import attribute_conflict_doc

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

hmeq_data=pd.read_csv('./HMEQ/hmeq_cleaned.csv',index_col=0)
# hmeq_data=hmeq_data.drop(labels=[0],axis=1)

'''
Removing Nan
'''
hmeq_data.dropna(inplace=True)

hmeq_label=hmeq_data.loc[:,'BAD']
hmeq_data=hmeq_data.drop(labels=['BAD'],axis=1)
'''
Applying PCA
'''
cov=(hmeq_data.T@hmeq_data)/(len(hmeq_data)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)
variance=np.zeros(len(hmeq_data.columns))
for i in range(len(hmeq_data.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
new_hmeq_data=hmeq_data@sorted_eig_vecs[:,:2]

'''
Performing KNN
'''

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_hmeq_data,hmeq_label,test_size=0.2,random_state=42)
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.subplot(1, 2, 1)
plt.title('With Outlier')
plt.plot(range(1,35),scores)


'''
Outlier Detection using DBSCAN
'''
i=1
while True:
    DBSCAN_model=DBSCAN(eps=i, min_samples=50).fit(new_hmeq_data)
    labels=DBSCAN_model.labels_
    i+=1
    print(len(np.where(labels==-1)[0]))
    if len(np.where(labels==-1)[0])<=0.055*len(new_hmeq_data) and len(np.where(labels==-1)[0])>=0.045*len(new_hmeq_data):
        print(len(np.where(labels==-1)[0]))
        print(i)
        break
hmeq_data_no_outlier=new_hmeq_data.loc[labels!=-1,:]
hmeq_label_no_outlier=hmeq_label.loc[labels!=-1]
# print(len(econometric_data))
# print(len(econometric_data_no_outlier))

'''
Performing KNN
'''

scores_no_outlier=list()
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(hmeq_data_no_outlier,hmeq_label_no_outlier,test_size=0.2,random_state=42)
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_no_outlier,y_train_no_outlier)
    y_pred=knn.predict(X_test_no_outlier)
    scores_no_outlier.append(metrics.accuracy_score(y_test_no_outlier,y_pred)) 
plt.subplot(1, 2, 2)
plt.title('Without Outlier')
plt.plot(range(1,35),scores_no_outlier)
plt.show()

plt.subplot(1, 2, 1)
plt.plot(new_hmeq_data.loc[:,0],new_hmeq_data.loc[:,1],'x')
plt.title('Points after PCA')

plt.subplot(1, 2, 2)
plt.plot(new_hmeq_data.loc[labels==-1,0],new_hmeq_data.loc[labels==-1,1],'x')
plt.title('Points removed')
plt.show()