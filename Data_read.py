import numpy as np
import pandas as pd

# Australian data
def aus():
    australian_dataset=pd.read_csv('./Datasets/uci-australian.dat',header=None,sep=' ')
    y_label=australian_dataset.loc[:,len(australian_dataset.columns)-1]
    X_data=australian_dataset.drop(australian_dataset.columns[[len(australian_dataset.columns)-1]],axis=1)
    min_outliers = 0.01
    max_outliers = 0.06
    Samples = 100
    #main(X_data,y_label)
    return X_data,y_label, min_outliers, max_outliers, Samples

# German data
def german():
    german_dataset=pd.read_csv('./Datasets/uci-german.csv',header=None)
    german_label=german_dataset.loc[:,len(german_dataset.columns)-1]
    german_dataset=german_dataset.drop(german_dataset.columns[[len(german_dataset.columns)-1]],axis=1)
    min_outliers = 0.02
    max_outliers = 0.04
    Samples = 50
    return german_dataset, german_label, min_outliers, max_outliers, Samples

# PAKDD data
def pakdd():
    pakdd_data=pd.read_csv('./Datasets/Pakdd_cleaned.csv',header=None)
    pakdd_label=pd.read_csv('./Datasets/Pakdd_labels_cleaned.csv',header=None)
    pakdd_label = pakdd_label.loc[:,len(pakdd_label.columns)-1]
    min_outliers = 0.01
    max_outliers = 0.03
    Samples = 10
    return pakdd_data, pakdd_label, min_outliers, max_outliers, Samples

# Thomas data
def thomas():    
    thomas_dataset=pd.read_csv('./Datasets/Thomas_cleaned.csv')
    thomas_label=thomas_dataset.loc[:,'BAD']
    thomas_dataset=thomas_dataset.drop(['BAD'],axis=1)
    min_outliers = 0.04
    max_outliers = 0.06
    Samples = 120
    return thomas_dataset, thomas_label, min_outliers, max_outliers, Samples

# Hmeq data
def hmeq():
    hmeq_data=pd.read_csv('./Datasets/hmeq_cleaned.csv',index_col=0)
    hmeq_data.dropna(inplace=True)
    hmeq_label=hmeq_data.loc[:,'BAD']
    hmeq_data=hmeq_data.drop(labels=['BAD'],axis=1)
    min_outliers = 0.045
    max_outliers = 0.055
    Samples = 50
    return hmeq_data, hmeq_label, min_outliers, max_outliers, Samples

# Japan data
def japan():
    japan_data=pd.read_csv('./Datasets/uci-japan-data-cleaned.csv',header=None)
    japan_label=pd.read_csv('./Datasets/uci-japan-labels-cleaned.csv',header=None)
    japan_label = japan_label.loc[:,len(japan_label.columns)-1]
    min_outliers = 0.04
    max_outliers = 0.055
    Samples = 10
    return japan_data,japan_label, min_outliers, max_outliers, Samples

# mortgage data
def mortgage():
    mortgage_data=pd.read_csv("./Datasets/cra-mortgage.csv")
    mortgage_label=mortgage_data.loc[:,'status_time'] - 1
    mortgage_data=mortgage_data.drop(['id','status_time'],axis=1)

    # morgage_data=morgage_data.drop(labels=['labels'],axis=1)
    min_outliers = 0.01
    max_outliers = 0.02
    Samples = 10
    return mortgage_data, mortgage_label, min_outliers, max_outliers, Samples

# econometric analysis data
def eco():
    econometric_data=pd.read_csv('./Datasets/eco_analysis_cleaned.csv')
    econometric_label=econometric_data.loc[:,'card']
    econometric_data=econometric_data.drop(labels=['card'],axis=1)
    min_outliers = 0.03
    max_outliers = 0.05
    Samples = 100
    return econometric_data, econometric_label, min_outliers, max_outliers, Samples