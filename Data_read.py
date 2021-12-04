import numpy as np
import pandas as pd

# Australian data
def aus():
    australian_dataset=pd.read_csv('./Datasets/uci-australian.dat',header=None,sep=' ')
    y_label=australian_dataset.loc[:,len(australian_dataset.columns)-1]
    X_data=australian_dataset.drop(australian_dataset.columns[[len(australian_dataset.columns)-1]],axis=1)
    #main(X_data,y_label)
    return X_data,y_label

# German data
def german():
    german_dataset=pd.read_csv('./Datasets/german_final.csv',header=None)
    german_label=german_dataset.loc[:,len(german_dataset.columns)-1]
    german_dataset=german_dataset.drop(german_dataset.columns[[len(german_dataset.columns)-1]],axis=1)
    return german_dataset, german_label

# PAKDD data
def pakdd():
    pakdd_data=pd.read_csv('./Datasets/Cleaned_Data.csv',header=None)
    pakdd_label=pd.read_csv('./Datasets/Data_labels.csv',header=None)
    pakdd_label = pakdd_label.loc[:,len(pakdd_label.columns)-1]
    return pakdd_data, pakdd_label

# Thomas data
def thomas():    
    thomas_dataset=pd.read_csv('./Datasets/Thomas_oneHot.csv')
    thomas_label=thomas_dataset.loc[:,'BAD']
    thomas_dataset=thomas_dataset.drop(['BAD'],axis=1)
    return thomas_dataset, thomas_label

# Hmeq data
def hmeq():
    hmeq_data=pd.read_csv('./Datasets/hmeq_cleaned.csv',index_col=0)
    hmeq_data.dropna(inplace=True)
    hmeq_label=hmeq_data.loc[:,'BAD']
    hmeq_data=hmeq_data.drop(labels=['BAD'],axis=1)
    return hmeq_data, hmeq_label

# Japan data
def japan():
    japan_data=pd.read_csv('./Datasets/uci-japan-data-cleaned.csv',header=None)
    japan_label=pd.read_csv('./Datasets/uci-japan-labels-cleaned.csv',header=None)
    japan_label = japan_label.loc[:,len(japan_label.columns)-1]
    return japan_data,japan_label

# mortgage data
def mortgage():
    morgage_data=pd.read_csv("./Datasets/mortgage_new.csv")
    morgage_data=morgage_data.drop(labels=['id'],axis=1)
    morgage_label=morgage_data.loc[:,'labels']
    morgage_data=morgage_data.drop(labels=['labels'],axis=1)
    return morgage_data, morgage_label

# econometric analysis data
def eco():
    econometric_data=pd.read_csv('./Datasets/eco_analysis.csv')
    econometric_label=econometric_data.loc[:,'card']
    econometric_data=econometric_data.drop(labels=['card'],axis=1)
    return econometric_data, econometric_label