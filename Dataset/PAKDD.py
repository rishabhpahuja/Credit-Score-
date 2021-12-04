import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import DBSCAN


pakdd_data=pd.read_csv('Cleaned_Data.csv',header=None)
print(pakdd_data)
# pakdd_data.drop(pakdd_data.columns[0],axis=1)
pakdd_label=pd.read_csv('Data_labels.csv',header=None)
pakdd_label = pakdd_label.loc[:,len(pakdd_label.columns)-1]
print(pakdd_label)
print("aaaaaaaaaaaaaaaaa",len(np.where(pakdd_label==1)[0]))

'''
Perform PCA
'''
print("zzzzzzzzzzzzzzzzzzzzzzzz")
#cov=(pakdd_data.T@pakdd_data)/(len(pakdd_data)-1)
cov = np.cov(pakdd_data.T,bias=False)
print("aaaaaaaaaaaaaaaaaaaaaa")
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

print("ppppppppppppppppppppppppppppppp")
variance=np.zeros(len(pakdd_data.columns))
for i in range(len(pakdd_data.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
# print(variance)

print("ccccccccccccccccccccccccccccccccccc")
new_pakdd_data=pakdd_data@sorted_eig_vecs[:,:2] #new data after PCA decomposition
# print(new_pakdd_data)
print(new_pakdd_data)


print('dddddddddddddddddddddddddddddddddddddddddddddddddd')
# k_means=KMeans(n_clusters=2)
# pred_data=k_means.fit_predict(new_pakdd_data)
# pred_data=k_means.fit_predict(pakdd_data)
# pred_data0=new_pakdd_data.loc[pred_data==0,:]
# pred_data1=new_pakdd_data.loc[pred_data==1,:]
# print(len(pred_data0))


# plt.scatter(pred_data0.loc[:,0],pred_data0.loc[:,1])
# plt.scatter(pred_data1.loc[:,0],pred_data1.loc[:,1])
# plt.show()

scores=list()
X_train,X_test,y_train,y_test=train_test_split(new_pakdd_data,pakdd_label,test_size=0.2,random_state=42)
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))

plt.subplot(1, 2, 1)
plt.title('With Outlier')
plt.plot(range(1,35),scores)

i=1
while True:
    DBSCAN_model=DBSCAN(eps=i, min_samples=10).fit(new_pakdd_data)
    labels=DBSCAN_model.labels_
    i+=1
    print(len(np.where(labels==-1)[0]))
    if len(np.where(labels==-1)[0])<=0.03*len(new_pakdd_data) and len(np.where(labels==-1)[0])>=0.01*len(new_pakdd_data):
        print(len(np.where(labels==-1)[0]))
        break

pakdd_datatset_no_outlier = new_pakdd_data.loc[labels!=-1,:]
pakdd_label_no_outlier = pakdd_label.loc[labels!=-1]
X_train_no_outlier,X_test_no_outlier,y_train_no_outlier,y_test_no_outlier=train_test_split(pakdd_datatset_no_outlier,pakdd_label_no_outlier,test_size=0.2,random_state=42)

scores_no_outlier=list()
for k in range(1,35):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_no_outlier,y_train_no_outlier)
    y_pred_no_outlier=knn.predict(X_test_no_outlier)
    scores_no_outlier.append(metrics.accuracy_score(y_test_no_outlier,y_pred_no_outlier))

plt.subplot(1, 2, 2)
plt.title('Without Outlier')
plt.plot(range(1,35),
scores_no_outlier)
plt.show()

plt.subplot(1, 2, 1)
a=np.where(labels==-1)
plt.plot(new_pakdd_data.loc[a,0],new_pakdd_data.loc[a,1],'x')

plt.subplot(1, 2, 2)
plt.plot(new_pakdd_data.loc[:,0],new_pakdd_data.loc[:,1],'x')
plt.show()