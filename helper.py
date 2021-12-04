import numpy as np
import pandas as pd

def mortgage():
    data = pd.read_csv('cra-mortgage.csv')
    data = data.drop(labels=['id','time'],axis=1)
    data = data.drop(labels=data.loc[data.loc[:,'status_time']==0,'status_time'])
    return(data)


if __name__ == '__main__':
    print(mortgage())