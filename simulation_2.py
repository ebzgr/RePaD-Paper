import numpy as np
import pandas as pd
import RePad.data_generator as dag
import RePad.discretization_generator as dig
import RePad.discretizer as dis
import RePad.utility as utl
import RePad.estimator as st
import simulation_utility as sutl


foldername = 'simulation2_results'

def attach_next(df, parts):
    data = utl.get_partitioning_variables(df)
    df['pi'] = utl.q_to_pi_states(parts, data['Q'], 30)
    df = df.rename(columns={'m':'x'})
    df = utl.add_next_state(df)
    
    return df

def check_probability_distribution(N, periodlist, true_partition=False):
    res = pd.DataFrame(columns = ['k', 'transition_sparsity', 'affect transition', 'bus periods', 'lamb', 'train_dec_pr', 'train_trans_pr', 'valid_dec_pr', 'valid_trans_pr'])

    for k in np.arange(N):
        print(k)
        if(true_partition):
            parts = pd.read_pickle("data/{}/partition_{}.pickle".format(foldername,k))
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in periodlist:
                    train_df = pd.read_pickle('data/{}/train_df_{}_{}_{}_{}.pickle'.format(foldername,k,ttype_ext,m_trans*1,periods))
                    valid_df = pd.read_pickle('data/{}/valid_df_{}_{}_{}_{}.pickle'.format(foldername,k,ttype_ext,m_trans*1,periods))
                    train_df = attach_next(train_df, parts)
                    valid_df = attach_next(valid_df, parts)
                    for lamb in np.arange(-5.5,-1,0.5):    
                        if(true_partition==False):
                            parts = pd.read_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(foldername, k,ttype_ext,m_trans*1,periods,lamb))
    
                        trans_pr, decision_pr = dis._get_prediction_tables(train_df, [0,1], 15, 10**lamb)

                        t_df = train_df.merge(trans_pr, how='left').merge(decision_pr, how='left')
                        v_df = valid_df.merge(trans_pr, how='left').merge(decision_pr, how='left')
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, np.sum(np.log(t_df.dec_pr)), np.sum(np.log(t_df.trans_pr)), np.sum(np.log(v_df.dec_pr)), np.sum(np.log(v_df.trans_pr))]

    tmp = res.groupby(['bus periods','transition_sparsity','affect transition','lamb'])[['train_dec_pr', 'train_trans_pr', 'valid_dec_pr', 'valid_trans_pr']].mean().reset_index()
    final = pd.pivot(tmp,index = ['bus periods','transition_sparsity','affect transition'],columns='lamb',values=['train_dec_pr', 'train_trans_pr', 'valid_dec_pr', 'valid_trans_pr'])
    
    return final

def simulation_analysis(n, dim_q, periodlist):
    res = pd.DataFrame(columns = ['k','transition_sparsity','affect transition','bus periods','lamb','score','false positive', 'false negative','similar'])
    for k in np.arange(n):
        print(k)
        partitions = pd.read_pickle("data/{}/partition_{}.pickle".format(foldername, k))
        for ttype_ext in [3,15]:
            for m_trans in [False,True]:
                for periods in periodlist:
                    for lamb in [0,0.2,0.5,1,2,5,100]:    
                        parts = pd.read_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(foldername, k,ttype_ext,m_trans*1,periods,lamb))
                        report = pd.read_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(foldername, k,ttype_ext,m_trans*1,periods,lamb))
                        score = report.loc[len(report)-1].test_score
                        false_positive, false_negative = sutl.bad_split_counter(partitions, parts, dim_q)
                        similar = len(sutl.find_similar_splits(partitions, parts))
                        res.loc[len(res)]=[k, ttype_ext, m_trans, periods, lamb, score, false_positive, false_negative, similar]
    
    res['false positive'] = res['false positive'].astype(int)
    res['false negative'] = res['false negative'].astype(int)
    res['similar'] = res['similar'].astype(int)
    tmp = res.groupby(['bus periods','transition_sparsity','affect transition','lamb'])[['score','false positive', 'false negative', 'similar']].mean().reset_index()
    final = pd.pivot(tmp,index = ['bus periods','transition_sparsity','affect transition'],columns='lamb',values=['score','false positive', 'false negative','similar'])
    return final
                        
                        

def single_run(data, val_data, k, ttype_ext, m_trans, periods, dim_pi, lamb):
    discretizer = dis.DataDriveDiscretizer(lamb = lamb, 
                            delta = 10**-6,
                            min_size = 1, 
                            smoothing_del = 10**-5,
                            max_pi = dim_pi)
    
    parts, report = discretizer.discretize(data, val_data, parallel=True)  
    parts.to_pickle("data/{}/parts_{}_{}_{}_{}_{}.pickle".format(foldername, k,ttype_ext,m_trans*1,periods,lamb))
    report.to_pickle("data/{}/report_{}_{}_{}_{}_{}.pickle".format(foldername, k,ttype_ext,m_trans*1,periods,lamb))


def simulation():
    dim_q = 30
    dim_active_q = 10
    max_q = 10
    dim_pi = 15
    max_pi = 15
    buses = 100
    start = 0
    n = 4
    par_gen = dig.RandomDiscretizationGenerator(dim_q, dim_active_q, dim_pi, max_q)
    for k in np.arange(start,n,4):
        partitions = par_gen.generate_random_discretization(1)
        partitions.to_pickle("data/{}/partition_{}.pickle".format(foldername,k))
        partitions['f_dc'] = 0
        for i in range(dim_active_q):
            partitions['f_dc'] = partitions['f_dc'] - (partitions['q_{}_min'.format(i)]+partitions['q_{}_max'.format(i)])/(2*dim_active_q)
        partitions['f_dc'] = (5 + partitions['f_dc'])*2-5
        
        for ttype_ext in [int(dim_pi/5),dim_pi]:
            q_transition = utl.generate_pi_transition(dim_pi, 3, ttype_ext)
            for m_trans in [False,True]:
                partitions['f_tr'] = 1 if m_trans else np.random.choice([0,1,2,3],dim_pi)

                
                data_gen = dag.EngineReplacementDataGenerator(max_mileage=20, mileage_coefficient=-0.2,
                                                    discretization=partitions, q_transition=q_transition, 
                                                    max_q=max_q, dim_q=dim_q, discounting_factor=0.9)
            
                for periods in [400,100]:
                    train_df = data_gen.generate(buses=buses, periods=periods)
                    data = utl.get_partitioning_variables(train_df)
                    train_df.to_pickle('data/{}/train_df_{}_{}_{}_{}.pickle'.format(foldername,k,ttype_ext,m_trans*1,periods))

                    valid_df = data_gen.generate(buses=buses, periods=periods)
                    val_data = utl.get_partitioning_variables(valid_df)
                    valid_df.to_pickle('data/{}/valid_df_{}_{}_{}_{}.pickle'.format(foldername,k,ttype_ext,m_trans*1,periods))
    
                    for lamb in [0,0.2,0.5,1,2,5,100]:
                        single_run(data, val_data, k, ttype_ext, m_trans, periods, dim_pi, lamb)

np.random.seed(0)
simulation()