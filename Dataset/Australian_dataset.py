import numpy as np
import pandas as pd
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

australian_dataset=pd.read_csv('./UCI Data/Australian/australian.dat',header=None,sep=' ')
print(australian_dataset)
australian_label=australian_dataset.loc[:,len(australian_dataset.columns)-1]
australina_dataset=australian_dataset.drop(australian_dataset.columns[[len(australian_dataset.columns)-1]],axis=1)

cov=(australian_dataset.T@australian_dataset)/(len(australian_dataset)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

variance=np.zeros(len(australian_dataset.columns))
for i in range(len(australian_dataset.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)


new_australian_dataset=australian_dataset@sorted_eig_vecs[:,:2]

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_australian_dataset,australian_label,test_size=0.2,random_state=42)
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k,leaf_size=2,p=2)
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
    DBSCAN_model=DBSCAN(eps=i, min_samples=5).fit(new_australian_dataset)
    labels=DBSCAN_model.labels_
    i+=1
    if len(np.where(labels==-1)[0])<=0.05*len(new_australian_dataset) and len(np.where(labels==-1)[0])>=0.04*len(new_australian_dataset):
        print(len(np.where(labels==-1)[0]))
        break

australian_datatset_no_outlier=new_australian_dataset.loc[labels!=-1,:]
australian_label_no_outlier=australian_label.loc[labels!=-1]
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(new_australian_dataset,australian_label,test_size=0.2,random_state=42)

scores_no_outlier=list()
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k,leaf_size=2,p=2)
    knn.fit(X_train_no_outlier,y_train_no_outlier)
    y_pred_no_outlier=knn.predict(X_test_no_outlier)
    scores_no_outlier.append(metrics.accuracy_score(y_test_no_outlier,y_pred_no_outlier))

plt.subplot(1, 2, 2)
plt.title('Without Outlier')
plt.plot(range(1,35),scores_no_outlier)
plt.show()

a=np.where(labels==-1)
plt.plot(new_australian_dataset.loc[a,0],new_australian_dataset.loc[a,1],'x')
plt.show()
# plt.plot(new_australian_dataset.loc[:,0],new_australian_dataset.loc[:,1],'x')
# plt.show()