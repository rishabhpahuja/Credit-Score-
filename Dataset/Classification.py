import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pakdd_data=pd.read_csv('./PAKDD/Cleaned_Data.csv',header=None)
print(pakdd_data)
# pakdd_data.drop(pakdd_data.columns[0],axis=1)
pakdd_label=pd.read_csv('./PAKDD/Data_labels.csv',header=None)
print(pakdd_label)
print("aaaaaaaaaaaaaaaaa",len(np.where(pakdd_label==1)[0]))

'''
Perform PCA
'''
print("zzzzzzzzzzzzzzzzzzzzzzzz")
cov=(pakdd_data.T@pakdd_data)/(len(pakdd_data)-1)
sorted_eig_vals,sorted_eig_vecs=np.linalg.eig(cov)

print("ppppppppppppppppppppppppppppppp")
variance=np.zeros(len(pakdd_data.columns))
for i in range(len(pakdd_data.columns)):
    variance[i]=sum(sorted_eig_vals[:i+1])/sum(sorted_eig_vals)
# print(variance)

print("ccccccccccccccccccccccccccccccccccc")
new_pakdd_data=pakdd_data@sorted_eig_vecs[:,:2] #new data after PCA decomposition
# print(new_pakdd_data)

'''

Perform PCA

'''
print('dddddddddddddddddddddddddddddddddddddddddddddddddd')
k_means=KMeans(n_clusters=2)
# pred_data=k_means.fit_predict(new_pakdd_data)
pred_data=k_means.fit_predict(pakdd_data)
pred_data0=new_pakdd_data.loc[pred_data==0,:]
pred_data1=new_pakdd_data.loc[pred_data==1,:]
print(len(pred_data0))
# plt.scatter(pred_data0.loc[:,0],pred_data0.loc[:,1])
# plt.scatter(pred_data1.loc[:,0],pred_data1.loc[:,1])
# plt.show()