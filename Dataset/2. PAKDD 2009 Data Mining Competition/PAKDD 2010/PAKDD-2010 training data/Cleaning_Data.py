import pandas as pd
import numpy as np
from pandas.io.sql import DatabaseError

orig_data=pd.read_csv('PAKDD2010_Modeling_Data.txt',encoding='latin1',sep='\t',header=None)
y_label=orig_data.loc[:,len(orig_data.columns)-1]
print(type(y_label))
orig_data=orig_data.drop(orig_data.columns[[1,10,11,12,13,14,15,34,35,36,51,52,53]],axis=1)
# orig_data.loc[:,3]=orig_data[3].replace(['Web','Carga'],[0,1])

assert(~(orig_data[4].isnull().any().any())) #check if there are any null values in column 4
# orig_data.loc[:,6]=orig_data[6].replace(['M','F'],[0,1])
orig_data.loc[:,16]=orig_data[16].replace(['Y','N'],[1,0])
orig_data.loc[:,18]=orig_data[18].replace(np.nan,0)
orig_data.loc[:,19]=orig_data[19].replace(np.nan,0)
orig_data.loc[:,20]=orig_data[20].replace(['Y','N'],[1,0])
print(orig_data[28].isnull().any().any())
orig_data.loc[:,33]=orig_data[33].replace(['Y','N'],[1,0])
orig_data.loc[:,37]=orig_data[37].replace(['Y','N'],[1,0])
# orig_data.loc[:,40]=orig_data[40].replace(np.nan,np.round((orig_data[40].sum())/50000))
# print("393939",orig_data[39].isnull().any().any())
# print(np.where(orig_data.loc[:,40]==0.0))
orig_data.loc[:,40]=orig_data[40].replace(np.nan,0.0)
orig_data.loc[:,41]=orig_data[41].replace(np.nan,0.0)
orig_data.loc[:,42]=orig_data[42].replace(np.nan,0.0)
orig_data.loc[:,43]=orig_data[43].replace(np.nan,0.0)
orig_data.loc[:,49]=orig_data[49].replace(['Y','N'],[1,0])
print("Empty cell",type(orig_data.loc[49999,17]))
orig_data.loc[:,17]=orig_data[17].replace(' ',0)
orig_data.loc[:,38]=orig_data[38].replace(' ',0)
orig_data.loc[:,5]=orig_data[5].replace(' ',0)
# print('sddddddddddddddddd',len((np.where(orig_data[38]==)[0])))

print('aaaaaaaaaaaaaaaaaaaaaaaa',np.where(orig_data==' '))
orig_data=pd.get_dummies(orig_data,columns=[3,6])
# print((orig_data[17].isnull().any().any()))

print(orig_data)
orig_data.to_csv('Cleaned_Data.csv')
y_label.to_csv('Data_labels.csv')
