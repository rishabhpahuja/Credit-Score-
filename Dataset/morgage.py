import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import DBSCAN

morgage_data=pd.read_csv("./Mortgage/mortgage_new.csv")
morgage_data=morgage_data.drop(labels=['id'],axis=1)
morgage_label=morgage_data.loc[:,'labels']
morgage_data=morgage_data.drop(labels=['labels'],axis=1)

'''
Applying PCA
'''
cov=(morgage_data.T@morgage_data)/(len(morgage_data)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)
variance=np.zeros(len(morgage_data.columns))
for i in range(len(morgage_data.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
new_morgage_data=morgage_data@sorted_eig_vecs[:,:2]
print(new_morgage_data)

'''
Performing KNN
'''

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_morgage_data,morgage_label,test_size=0.2,random_state=42)
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
i=5000
while True:
    DBSCAN_model=DBSCAN(eps=i, min_samples=10).fit(new_morgage_data)
    labels=DBSCAN_model.labels_
    i+=5
    print(len(np.where(labels==-1)[0]))
    if len(np.where(labels==-1)[0])<=0.02*len(new_morgage_data) and len(np.where(labels==-1)[0])>=0.01*len(morgage_label):
        print(len(np.where(labels==-1)[0]))
        print(i)
        break
morgage_data_no_outlier=new_morgage_data.loc[labels!=-1,:]
morgage_label_no_outlier=morgage_label.loc[labels!=-1]
# print(len(morgage_data))
# print(len(econometric_data_no_outlier))

'''
Performing KNN
'''

scores_no_outlier=list()
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(morgage_data_no_outlier,morgage_label_no_outlier,test_size=0.2,random_state=42)
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
plt.plot(new_morgage_data.loc[:,0],new_morgage_data.loc[:,1],'x')
plt.title('Points after PCA')

plt.subplot(1, 2, 2)
plt.plot(new_morgage_data.loc[labels==-1,0],new_morgage_data.loc[labels==-1,1],'x')
plt.title('Points removed')
plt.show()