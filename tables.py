import numpy as np
import pandas as pd
import simulation_utility as sutl
import matplotlib.pyplot as plt
from collections import Counter

types = ['No transition','Random transition', 'Sparse transition']

def sim1_alt_result():
    res1 = pd.DataFrame(columns = ['replacement cost','transit mil', 'transition type',1,2,4,6])
    res2 = res1.copy()
    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                df = pd.read_pickle('data/simulation1_results/res_{}_{}_{}.pickle'.format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))[[2,5,9,15]]
                df = pd.DataFrame(data = np.concatenate(df.values.flatten()).reshape(-1,4))
                res1.loc[len(res1)] = [diff_rep_cost*1, ml_tr_mode*1, q_trans_mod] + df.mean().round(3).tolist()
                
                s = ['','','']
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
    combined = combined.replace(True,'Diverse')
    combined = combined.replace(False,'Similar')
                
    return combined.to_latex(index=False)

def simple_simulation_res_alt():
    res1 = pd.DataFrame(columns = ['replacement cost','transit mil', 'transition type','f0','f1','f2','f3','tr_f'])
    res2 = res1.copy()
    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                df = pd.read_pickle('data/sim1_alt/res_{}_{}_{}.pickle'.format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))[[10,11,12,13,3]]
#                df = pd.DataFrame(data = np.concatenate(df.values.flatten()).reshape(-1,7))
                res1.loc[len(res1)] = [diff_rep_cost*1, ml_tr_mode*1, q_trans_mod] + df.mean().round(3).tolist()
                
                s = ['','','']
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
    combined = combined.replace(True,'Diverse')
    combined = combined.replace(False,'Similar')
        
    return combined.to_latex(index=False)

def simple_simulation_res():
    df = pd.read_pickle('data/simulation_results.pickle')
    df = df.reset_index().rename(columns={'index':'variable'})
    tmp = pd.pivot(df,index = ['replacement cost','transit mil', 'transition type'], columns=['variable'], values=['mean','2%','98%'])
    tmp = tmp.swaplevel(0, 1, 1).sort_index(1).fillna('---')
    tmp = tmp.replace('---',0)
    
    res1 = tmp.reset_index().iloc[:,:3]
    res2 = tmp.reset_index().iloc[:,:3]

    for var in ['alpha','f0','f1','f2','f3','tr_alpha','tr_f']:
        if('alpha' in var):
            res1[var] = tmp[(var,'mean')].round(3).astype(str).tolist()
            res2[var] = (' ('+ tmp[(var,'2%')].round(3).astype(str)+', ' + tmp[(var,'98%')].round(3).astype(str) +')').tolist()            
        else:
            res1[var] = tmp[(var,'mean')].round(2).astype(str).tolist()
            res2[var] = (' ('+ tmp[(var,'2%')].round(2).astype(str)+', ' + tmp[(var,'98%')].round(2).astype(str) +')').tolist()
        
    combined = pd.DataFrame(columns = res1.columns)


    
    for i in range(len(res1)):
        combined.loc[len(combined)] = res1.loc[i]
        combined.loc[len(combined)] = ['','','']+res2.iloc[i][3:].tolist()
    
    for i in range(3):
        combined['transition type'] = combined['transition type'].replace(i+1,types[i])

    
    combined = combined.replace('0.0','---')
    combined = combined.replace(' (0.0, 0.0)','---')
    combined = combined.replace(True,'Diverse')
    combined = combined.replace(False,'Similar')
        
    return combined.to_latex(index=False)

def simulation_2_res_alt(N, dim_q, expname):
    res = pd.DataFrame(columns = ['k','transition_sparsity','affect transition','bus periods','lamb','metric','value'])
    for k in np.arange(N):
        print(k)
        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(expname, k))
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in [100,400]:
                    for lamb in [0,0.2,0.5,1,2,5,100]:    
                        parts = pd.read_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        report = pd.read_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        score = report.loc[len(report)-1].score
                        false_positive, false_negative = sutl.bad_split_counter(partitions, parts, dim_q)
                        similar = len(sutl.find_similar_splits(partitions, parts))
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'vscore', score]
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'false negative', false_negative]
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'false positive', false_positive]
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'similar', similar]
                        
    
    res['value'] = res['value'].astype(float)

    tmp = res.groupby(['metric','bus periods','transition_sparsity','affect transition','lamb'])['value'].mean().reset_index()
    final = pd.pivot(tmp,index = ['metric','bus periods','transition_sparsity','affect transition'],columns='lamb',values=['value']).reset_index()
    final.value = final.value.round(2)
    final.loc[final.metric=='vscore','value']= final[final.metric=='vscore'].value.round(0).values
    
    final = final.sort_values(by=['metric','bus periods','transition_sparsity','affect transition'],ascending = [False,True,True,True])
    final.transition_sparsity =  final.transition_sparsity.replace([3,15],['Sparse','Random'])
    final['affect transition'] = final['affect transition'].replace([False,True],['Yes','No'])
    
    return final.to_latex(index=False)

def simulation_2_res(N, dim_q, expname):
    df = {}
    for var in ['score','false_pos','false_neg','similar']:
        df[var] = pd.DataFrame(columns = ['k','transition_sparsity','affect transition','bus periods','lamb','value'])

    for k in np.arange(N):
        print(k)
        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(expname, k))
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in [100,400]:
                    for lamb in [0,0.2,0.5,1,2,5,100]:    
                        parts = pd.read_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        report = pd.read_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        score = report.loc[len(report)-1].test_score
                        false_positive, false_negative = sutl.bad_split_counter(partitions, parts, dim_q)
                        similar = len(sutl.find_similar_splits(partitions, parts))
                        df['score'].loc[len(df['score'])]=[k, ttype_ext, m_trans, periods, lamb, score]
                        df['false_neg'].loc[len(df['false_neg'])]=[k, ttype_ext, m_trans, periods, lamb, false_negative]
                        df['false_pos'].loc[len(df['false_pos'])]=[k, ttype_ext, m_trans, periods, lamb, false_positive]
                        df['similar'].loc[len(df['similar'])]=[k, ttype_ext, m_trans, periods, lamb, similar]
                        
    s = []
    for var in ['score','false_pos','false_neg','similar']:    
        df[var]['value'] = df[var]['value'].astype(float)
        tmp = df[var].groupby(['bus periods','transition_sparsity','affect transition','lamb'])['value'].mean().reset_index()
        if(var == 'score'):
            tmp['value'] = tmp['value'].astype(int)
        else:
            tmp['value'] = tmp['value'].round(2)
        final = pd.pivot(tmp,index = ['bus periods','transition_sparsity','affect transition'],columns='lamb',values=['value']).reset_index()
        final.transition_sparsity =  final.transition_sparsity.replace([3,15],['Sparse','Random'])
        final['affect transition'] = final['affect transition'].replace([False,True],['Yes','No'])
        s.append(final.to_latex(index=False))
    
    return s


def simulation_2_res_ll_only(N, dim_q, expname):
    res = pd.DataFrame(columns = ['k','transition_sparsity','affect transition','bus periods','lamb','metric','value'])
    for k in np.arange(N):
        print(k)
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in [100,400]:
                    for lamb in [0,0.2,0.5,1,2,5,100]:    
                        report = pd.read_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(expname, k,ttype_ext,m_trans*1,periods,lamb))
                        score = report.loc[len(report)-1].score
                        test_score = report.loc[len(report)-1].test_score
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'score', score]
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, 'test score', test_score]
                        
    res['value'] = res['value'].astype(float)
    tmp = res.groupby(['metric','bus periods','transition_sparsity','affect transition','lamb'])['value'].mean().reset_index()
    final = pd.pivot(tmp,index = ['metric','bus periods','transition_sparsity','affect transition'],columns='lamb',values=['value']).reset_index()
    final.value = final.value.round(0)
    
    final = final.sort_values(by=['metric','bus periods','transition_sparsity','affect transition'],ascending = [False,True,True,True])
    final.transition_sparsity =  final.transition_sparsity.replace([3,15],['Sparse','Random'])
    final['affect transition'] = final['affect transition'].replace([False,True],['Yes','No'])
    
    return final.to_latex(index=False)

def total_split_table(N, dim_active_q, expname):
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
     
    fig, ax = plt.subplots()
    ax.hist(f_dc)
    fig.savefig("f_dc_hist.png")
    
    counter = Counter(splits)
    x = np.arange(min(splits), max(splits)+1)
    y = [counter[i]/len(splits) for i in x]

    fig, ax = plt.subplots()
    ax.bar(x,y)
    fig.savefig("total_split_hist.png")        
    
    return splits, f_dc

print(sim1_alt_result())