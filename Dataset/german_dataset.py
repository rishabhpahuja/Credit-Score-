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
# # german_dataset=pd.read_csv(,header=None,encoding='latin1')
# for l in line:

    
#     writer_object.writerow(l.split())
# file.close()
# f.close()
# german_label=german_dataset.loc[:,len(german_dataset.columns)-1]
# print(german_dataset)

german_dataset=pd.read_csv('./UCI Data/German/german_final.csv',header=None)
german_label=german_dataset.loc[:,len(german_dataset.columns)-1]
german_dataset=german_dataset.drop(german_dataset.columns[[len(german_dataset.columns)-1]],axis=1)

'''
Perform PCA
'''

cov=(german_dataset.T@german_dataset)/(len(german_dataset)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

variance=np.zeros(len(german_dataset.columns))
for i in range(len(german_dataset.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
print(variance)

new_german_dataset=german_dataset@sorted_eig_vecs[:,:2]


'''
Since first two values consists of 97% variance, we reduce dimeniosn to two
'''

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_german_dataset,german_label,test_size=0.2,random_state=42)
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
    DBSCAN_model=DBSCAN(eps=i, min_samples=5).fit(german_dataset)
    labels=DBSCAN_model.labels_
    i+=1
    if len(np.where(labels==-1)[0])<=0.055*len(german_dataset) and len(np.where(labels==-1)[0])>=0.045*len(german_dataset):
        print(len(np.where(labels==-1)[0]))
        break
# german_dataset_no_outlier=new_german_dataset.loc[np.where(labels!=-1),:]
# print(np.where(labels!=-1))
# print("55555555555555",len(np.where(labels=-1)[0]))
german_datatset_no_outlier=new_german_dataset.loc[labels!=-1,:]
german_label_no_outlier=german_label.loc[labels!=-1]
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(new_german_dataset,german_label,test_size=0.2,random_state=42)

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