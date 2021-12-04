import numpy as np
import pandas as pd
from csv import writer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# f= open('german_final.csv','a')
# file=open('./UCI Data/German/german_data_numeric.txt','r')
# print(file)
# line=file.readlines()
# writer_object=writer(f)
# # thomas_dataset=pd.read_csv(,header=None,encoding='latin1')
# for l in line:

    
#     writer_object.writerow(l.split())
# file.close()
# f.close()
# thomas_label=thomas_dataset.loc[:,len(thomas_dataset.columns)-1]
# print(thomas_dataset)

thomas_dataset=pd.read_csv('Thomas_oneHot.csv',header=None)
thomas_label=thomas_dataset.loc[:,'BAD']
thomas_dataset=thomas_dataset.drop(['BAD'],axis=1)

'''
Perform PCA
''' 

cov=(thomas_dataset.T@thomas_dataset)/(len(thomas_dataset)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

variance=np.zeros(len(thomas_dataset.columns))
for i in range(len(thomas_dataset.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
print(variance)

new_thomas_dataset=thomas_dataset@sorted_eig_vecs[:,:2]


'''
Since first two values consists of 97% variance, we reduce dimeniosn to two
'''

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_thomas_dataset,thomas_label,test_size=0.2,random_state=42)
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
    DBSCAN_model=DBSCAN(eps=i, min_samples=100).fit(new_thomas_dataset)
    labels=DBSCAN_model.labels_
    i+=1
    print(len(np.where(labels==-1)[0]))
    if len(np.where(labels==-1)[0])<=0.05*len(thomas_dataset) and len(np.where(labels==-1)[0])>=0.03*len(thomas_dataset):
        print(len(np.where(labels==-1)[0]))
        break
# german_dataset_no_outlier=new_thomas_dataset.loc[np.where(labels!=-1),:]
# print(np.where(labels!=-1))
# print("55555555555555",len(np.where(labels=-1)[0]))
thomas_datatset_no_outlier=new_thomas_dataset.loc[labels!=-1,:]
thomas_label_no_outlier=thomas_label.loc[labels!=-1]
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(thomas_datatset_no_outlier,thomas_label_no_outlier,test_size=0.2,random_state=42)

scores_no_outlier=list()
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_no_outlier,y_train_no_outlier)
    y_pred_no_outlier=knn.predict(X_test_no_outlier)
    scores_no_outlier.append(metrics.accuracy_score(y_test_no_outlier,y_pred_no_outlier))

cov=(thomas_datatset_no_outlier.T@thomas_datatset_no_outlier)/(len(thomas_datatset_no_outlier)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

variance=np.zeros(len(thomas_datatset_no_outlier.columns))
for i in range(len(thomas_datatset_no_outlier.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
print(variance)

new_thomas_dataset=thomas_dataset@sorted_eig_vecs[:,:2]


plt.subplot(1, 2, 2)
plt.title('Without Outlier')
plt.plot(range(1,35),
scores_no_outlier)
plt.show()

plt.subplot(1, 2, 1)
a=np.where(labels==-1)
plt.plot(new_thomas_dataset.loc[a,0],new_thomas_dataset.loc[a,1],'x')

plt.subplot(1, 2, 2)
plt.plot(new_thomas_dataset.loc[:,0],new_thomas_dataset.loc[:,1],'x')
plt.show()