import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", palette="Set2")
data = pd.read_csv('accuracy.csv',header=None)
datasets = set(data.iloc[:,2])
for i in datasets:
    new_data = data.loc[data.loc[:,2]==i,:]
    g = sns.catplot(data=new_data,kind='bar',x=0,y=1,hue=3,legend=False)
    g.set(xlabel="Method",ylabel="Accuracy ",title=str('Dataset:'+i))
    g.set_xticklabels(rotation=30)
    plt.legend(loc='upper right')
    plt.savefig(str('accuracy_plots/'+i+'.png'),bbox_inches='tight')
    #plt.show()

