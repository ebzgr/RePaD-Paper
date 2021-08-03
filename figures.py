import numpy as np
import pandas as pd
import utility as utl
import pdb
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['axes.labelsize'] = (45)
plt.rcParams.update({'font.size': 45})
plt.rcParams["text.usetex"] = "False"
plt.rcParams["legend.loc"] = 'lower right'
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.linewidth'] = 2

def total_split_figure(N, dim_active_q, expname):
    splits = []
    f_dc = []
    for k in np.arange(N):
        print(k)

        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(expname, k))
        splits.extend(partitions.total_split.values.tolist())
        
        partitions['f_dc'] = 0
        for i in range(dim_active_q):
            partitions['f_dc'] = partitions['f_dc'] - (partitions['q_{}_min'.format(i)]+partitions['q_{}_max'.format(i)])/(2*dim_active_q)
        partitions['f_dc'] = (5 + partitions['f_dc'])*2-5
        
        f_dc.extend(partitions.f_dc.values.tolist())
     
    fig, [ax1,ax2] = plt.subplots(1,2,figsize=(30,10))
    ax1.hist(f_dc,density=True)
    ax1.set_xlabel("Replacement Cost")
    ax1.set_ylabel("Ratio")
#    fig.savefig("data/plots/f_dc_hist.png")
    
    counter = Counter(splits)
    x = np.arange(min(splits), max(splits)+1)
    y = [counter[i]/len(splits) for i in x]

#    fig, ax = plt.subplots()
    ax2.bar(x,y)
    ax2.set_xlabel("Number of Splits")

    fig.savefig("data/plots/partitionings.png")        
    
    return splits, f_dc