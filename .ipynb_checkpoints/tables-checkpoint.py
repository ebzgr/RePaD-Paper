import numpy as np
import pandas as pd
import simulation_utility as sutl
import matplotlib.pyplot as plt
from collections import Counter

types = ['No transition','Random transition', 'Sparse transition']

def table_1(latex=False):
    res1 = pd.DataFrame(columns = ['Case No','replacement cost','transit mil', 'transition type',1,2,4,6])
    res2 = res1.copy()
    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                df = pd.read_pickle('data/simulation1_results/res_{}_{}_{}.pickle'.format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))[[2,5,9,15]]
                df = pd.DataFrame(data = np.concatenate(df.values.flatten()).reshape(-1,4))
                res1.loc[len(res1)] = ["Case {}".format(len(res1)+1),diff_rep_cost*1, ml_tr_mode*1, q_trans_mod] + df.mean().round(3).tolist()
                s = ['','','','']
                quantiles = df.quantile([0.02,0.98]).round(3)
                for i in range(4):
                    s.append("({}, {})".format(quantiles[i][0.02],quantiles[i][0.98]))
                res2.loc[len(res2)] =  s

    combined = pd.DataFrame(columns = res1.columns)
    for i in range(len(res1)):
        combined.loc[len(combined)] = res1.loc[i]
        combined.loc[len(combined)] = res2.loc[i]  
        
    for i in range(3):
        combined['transition type'] = combined['transition type'].replace(i+1,types[i])

    
    combined = combined.replace('0.0','---')
    combined = combined.replace(' (0.0, 0.0)','---')
    combined = combined.replace(True,'Dissimilar')
    combined = combined.replace(False,'Similar')
    if(latex):            
        return combined.to_latex(index=False,float_format="%.3f")
    return combined

def table_2(latex=False):
    res1 = pd.DataFrame(columns = ['Case No','replacement cost','transit mil', 'transition type','f0','f1','f2','f3','c_r'])
    res2 = res1.copy()
    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                df = pd.read_pickle('data/simulation1_results/res_{}_{}_{}.pickle'.format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))[[10,11,12,13,3]]
                res1.loc[len(res1)] = ["Case {}".format(len(res1)+1),diff_rep_cost*1, ml_tr_mode*1, q_trans_mod] + df.mean().round(3).tolist()
                
                s = ['','','','']
                quantiles = df.quantile([0.02,0.98]).round(3)
                for i in [10,11,12,13,3]:
                    s.append("({}, {})".format(quantiles[i][0.02],quantiles[i][0.98]))
                res2.loc[len(res2)] =  s

    combined = pd.DataFrame(columns = res1.columns)
    for i in range(len(res1)):
        combined.loc[len(combined)] = res1.loc[i]
        combined.loc[len(combined)] = res2.loc[i]
        
        
    for i in range(3):
        combined['transition type'] = combined['transition type'].replace(i+1,types[i])

    
    combined = combined.replace('0.0','---')
    combined = combined.replace(' (0.0, 0.0)','---')
    combined = combined.replace(True,'Dissimilar')
    combined = combined.replace(False,'Similar')
        
    if(latex):            
        return combined.to_latex(index=False, float_format="%.3f")
    return combined

def table_3(N, dim_q, latex=False, expname='simulation2_results'):
    df = pd.DataFrame(columns = ['k','transition_sparsity','affect transition','bus periods','lamb','value'])

    for k in np.arange(N):
        print("Reading results of replication {}".format(k))
        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(expname, k))
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in [100,400]:
                    for lamb in [0,0.2,0.5,1,2,5,100]:    
                        parts = pd.read_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        report = pd.read_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        score = report.loc[len(report)-1].test_score
                        df.loc[len(df)]=[k, ttype_ext, m_trans, periods, lamb, score]
                        
 
    df['value'] = df['value'].astype(float)
    tmp = df.groupby(['bus periods','transition_sparsity','affect transition','lamb'])['value'].mean().reset_index()
    final = pd.pivot(tmp,index = ['bus periods','transition_sparsity','affect transition'],columns='lamb',values=['value']).reset_index()
    final.transition_sparsity =  final.transition_sparsity.replace([3,15],['Sparse','Random'])
    final['affect transition'] = final['affect transition'].replace([False,True],['Yes','No'])
    final['Case No'] = ['Case {}'.format(i) for i in range(1,len(final)+1)]
    last_column = final.columns[-1]
    new_columns_order = [last_column] + [col for col in final.columns if col != last_column]

    print(new_columns_order)
    final = final[new_columns_order]
    if(latex):
        return final.to_latex(index=False, float_format="%.0f")
    return final
    

def figure_a3(N, dim_active_q, expname='simulation2_results'):
    splits = []
    f_dc = []
    for k in np.arange(N):
        print("Reading results of replication {}".format(k))
        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(expname, k))
        splits.extend(partitions.total_split.values.tolist())
        
        partitions['f_dc'] = 0
        for i in range(dim_active_q):
            partitions['f_dc'] = partitions['f_dc'] - (partitions['q_{}_min'.format(i)] + partitions['q_{}_max'.format(i)]) / (2 * dim_active_q)
        partitions['f_dc'] = (5 + partitions['f_dc']) * 2 - 5
        f_dc.extend(partitions.f_dc.values.tolist())
     
    # First plot
    fig1, ax1 = plt.subplots()
    ax1.hist(f_dc)
    fig1.savefig("data/f_dc_hist.png")

    # Second plot
    counter = Counter(splits)
    x = np.arange(min(splits), max(splits) + 1)
    y = [counter[i] / len(splits) for i in x]

    fig2, ax2 = plt.subplots()
    ax2.bar(x, y)
    fig2.savefig("data/total_split_hist.png")        


    return fig1, fig2  # Return the figure objects