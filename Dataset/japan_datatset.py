import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import append
import pandas as pd
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

japan_dataset=pd.read_csv('./UCI Data/Japan/crx.data',header=None,sep=',')




# print(japan_dataset)
# print(len(np.where(japan_dataset.loc[:,14]=='')[0]))

index=list()
for i in range(len(japan_dataset.columns)):
    empty_indx=np.where(japan_dataset.loc[:,i]=='?')
    index=index+empty_indx[0].tolist()

remove=list()
for i in range(len(index)):
    if index[i] not in remove:
        remove.append(index[i])

japan_dataset=japan_dataset.drop(labels=remove,axis=0)
japan_label=japan_dataset.loc[:,len(japan_dataset.columns)-1]
japan_dataset=japan_dataset.drop(japan_dataset.columns[[len(japan_dataset.columns)-1]],axis=1)
japan_label=japan_label.replace(['+','-'],[1,0])
japan_label.to_csv('Japan_label.csv',index=False,header=False)





# Performing one hot encoding for categorical classification
japan_dataset=pd.get_dummies(japan_dataset,columns=[0,3,4,5,6,8,9,11,12])
japan_dataset.loc[:,1]=pd.to_numeric(japan_dataset.loc[:,1], downcast="float")
japan_dataset.loc[:,13]=japan_dataset.loc[:,13].astype(int)
japan_dataset.to_csv('Japan_Data.csv',index=False,header=False)

# for i in (japan_dataset.columns):
#     if japan_dataset[i].dtype == str:
#         print('yes')
#     else:
#         print('no')
print(japan_dataset.dtypes)
print(japan_dataset)

'''
Applying PCA
'''
cov=(japan_dataset.T@japan_dataset)/(len(japan_dataset)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

variance=np.zeros(len(japan_dataset.columns))
for i in range(len(japan_dataset.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)

# print(sorted_eig_vecs)

new_japan_dataset=japan_dataset@sorted_eig_vecs[:,:2]
print(new_japan_dataset)

'''
Performing KNN
'''

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_japan_dataset,japan_label,test_size=0.2,random_state=42)
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.subplot(1, 2, 1)
plt.title('With Outlier')
plt.plot(range(1,35),scores)

'''
Removing Outliers using DBSCAN
'''

i=5
while True:
    DBSCAN_model=DBSCAN(eps=i, min_samples=10).fit(new_japan_dataset)
    labels=DBSCAN_model.labels_
    i+=5
    print(len(np.where(labels==-1)[0]))
    if len(np.where(labels==-1)[0])<=0.055*len(new_japan_dataset) and len(np.where(labels==-1)[0])>=0.04*len(japan_label):
        # print(len(np.where(labels==-1)[0]))
        print(i)
        break
japan_dataset_no_outlier=new_japan_dataset.loc[labels!=-1,:]
japan_label_no_outlier=japan_label.loc[labels!=-1]

'''
Performing KNN
'''

scores_no_outlier=list()
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(japan_dataset_no_outlier,japan_label_no_outlier,test_size=0.2,random_state=42)
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
plt.plot(new_japan_dataset.loc[:,0],new_japan_dataset.loc[:,1],'x')
plt.title('Points after PCA')

plt.subplot(1, 2, 2)
plt.plot(new_japan_dataset.loc[labels==-1,0],new_japan_dataset.loc[labels==-1,1],'x')
plt.title('Points removed')
plt.show()

