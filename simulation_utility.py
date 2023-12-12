import pandas as pd
import numpy as np


def get_simulation_statistics(sim_res):
    df = None
    sim_res.tr_alpha = -1*sim_res.tr_alpha
    sim_res.alpha = -1*sim_res.alpha
    for var in ['tr_alpha','tr_f','alpha','f','pi_len']:
        if(var == 'f'):
            mx = pd.DataFrame(sim_res.f.tolist()).values
            stats = pd.DataFrame(mx).describe(percentiles=[0.02,0.98])
            stats = stats.add_prefix(var).T
        elif(var == 'pi_len'):
            stats = pd.DataFrame(sim_res.pi_len).describe(percentiles=[0.02,0.98])
            stats.columns = [var]
            stats = stats.T
        else:
            mx = np.concatenate(sim_res[var]).reshape(len(sim_res),-1)
            stats = pd.DataFrame(mx).describe(percentiles=[0.02,0.98])
            stats.columns = [var]
            stats = stats.T
        if(df is None):
            df = stats
        else:
            df = df.append(stats)
        
    return df

def simul_analysis():
    df = None
    for q_trans_mod in [1,2,3]:
        for ml_tr_mode in [True,False]:
            for diff_rep_cost in [True,False]:
                sim_res = pd.read_pickle("data/Monte_mode_{}_diffx_{}_diffcost_{}.pickle".format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))
                tmp = get_simulation_statistics(sim_res)
                tmp['transition type'] = q_trans_mod
                tmp['transit mil'] = ml_tr_mode
                tmp['replacement cost'] = diff_rep_cost
                if(df is None):
                    df = tmp
                else:
                    df = df.append(tmp)
                    
    df.to_pickle('data/simulation_results.pickle')
    
    return df

def get_partitioning_variables(df):
    df=df.rename(columns={'m':'x0'})
    ids = df.id.values
    periods = df.t.values
    X = df[['x0']].values
    Q = df[df.columns[df.columns.str.contains('q_')]].values
    Y = df.d.values
    return {'ids':ids, 'periods':periods, 'X':X, 'Q':Q, 'Y':Y}

def find_similar_splits(partitions, parts):
    cols = [col for col in partitions.columns if 'q_' in col]
    
    comparison_df = partitions[cols].merge(parts[cols], indicator=True, how='outer')
    same_df = comparison_df[comparison_df['_merge'] == 'both']
    
    return same_df

def bad_split_counter(partitions, parts, dim_q):
    false_positive = 0
    false_negative = 0
    
    for q in range(dim_q):
        orig_splitted = (partitions['q_{}_min'.format(q)]!=0).any()
        alg_splitted = (parts['q_{}_min'.format(q)]!=0).any()
        if((orig_splitted==False) and (alg_splitted==True)):
            false_positive += 1
        if((orig_splitted==True) and (alg_splitted==False)):
            false_negative += 1
            
    return false_positive, false_negative