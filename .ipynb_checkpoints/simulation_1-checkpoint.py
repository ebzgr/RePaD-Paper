import numpy as np
import pandas as pd
import RePad.data_generator as dag
import RePad.discretizer as dis
import RePad.utility as utl
import RePad.estimator as st
from joblib import Parallel, delayed
import time

foldername = 'simulation1_results'

def simulation_1_single_run(data_gen, buses, periods, dim_q, q_trans_mod, ml_tr_mode, diff_rep_cost, k):
    train_df = data_gen.generate(buses=buses, periods=periods)
    data = utl.get_partitioning_variables(train_df)
    train_df.to_pickle('data/{}/train_{}_{}_{}_{}.pickle'.format(foldername, q_trans_mod, ml_tr_mode*1, diff_rep_cost*1, k))

    test_df = data_gen.generate(buses=buses, periods=periods)
    test_data = utl.get_partitioning_variables(test_df)
    test_df.to_pickle('data/{}/test_{}_{}_{}_{}.pickle'.format(foldername, q_trans_mod, ml_tr_mode*1, diff_rep_cost*1, k))

    res = []
    
    for max_pi in [1,2,4,6]:
        estimator = st.BusEngineNFXP()
        discretizer = dis.DataDriveDiscretizer(delta=0, max_pi=max_pi)    
        parts, report = discretizer.discretize(data = data, parallel=False)

        parts.to_pickle('data/{}/parts_{}_{}_{}_{}.pickle'.format(foldername, q_trans_mod, ml_tr_mode*1, diff_rep_cost*1, k))
        report.to_pickle('data/{}/report_{}_{}_{}_{}.pickle'.format(foldername, q_trans_mod, ml_tr_mode*1, diff_rep_cost*1, k))


        pi = utl.q_to_pi_states(parts, test_data['Q'], dim_q)
        bestll, f, alpha = estimator.estimate_theta(test_data['ids'], test_data['periods'], test_data['X'].flatten().tolist(), pi, test_data['Y'])
        

        f.sort()
        res.extend([bestll,alpha])
        res.extend(f)
        
    return res

def monte_carlo(data_gen, buses, periods, dim_q, N, q_trans_mod, ml_tr_mode, diff_rep_cost, parallel=True):
    df = pd.DataFrame(columns=np.arange(1,22))
    if(parallel):
        element_run = Parallel(n_jobs=-1)(delayed(simulation_1_single_run)(data_gen, buses, periods, dim_q, q_trans_mod, ml_tr_mode, diff_rep_cost, k) for k in range(N))
        for i in range(N):
            df.loc[len(df)]=element_run[i]
    else:
        for k in range(N):
            res = simulation_1_single_run(data_gen, buses, periods, dim_q, q_trans_mod, ml_tr_mode, diff_rep_cost, k)
            df.loc[len(df)] = res
        
    df.to_pickle('data/{}/res_{}_{}_{}.pickle'.format(foldername, q_trans_mod, ml_tr_mode*1, diff_rep_cost*1)) 


def simulation():
    dim_q = 10
    max_q = 10
    buses = 400
    periods = 100
    N = 100
    mileage_coefficient = -0.2

    
    # Generating the data given a partitioning
    partitions = pd.DataFrame(columns = ['state','q_0_min','q_0_max','q_1_min','q_1_max'])
    partitions.loc[len(partitions)] = [0,0,5,0,5]
    partitions.loc[len(partitions)] = [1,0,5,5,10]
    partitions.loc[len(partitions)] = [2,5,10,5,10]
    partitions.loc[len(partitions)] = [3,5,10,0,5]
    for i in range(2,dim_q):
        partitions['q_{}_min'.format(i)] = 0
        partitions['q_{}_max'.format(i)] = 10

    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                if(diff_rep_cost):
                    partitions['f_dc'] = np.arange(4,8)*-1
                else:
                    partitions['f_dc'] = np.ones(4)*-5
                    
                if(ml_tr_mode):
                    partitions['f_tr'] = np.arange(4)
                else:
                    partitions['f_tr'] = np.ones(4)       
                    
                data_gen = dag.EngineReplacementDataGenerator(max_mileage=20, mileage_coefficient=mileage_coefficient, 
                                                    discretization=partitions, q_transition=q_trans_mod, 
                                                    max_q=max_q, dim_q=dim_q, discounting_factor=0.9)
                monte_carlo(data_gen, buses, periods, dim_q, N, q_trans_mod, ml_tr_mode, diff_rep_cost, parallel=True)


#np.random.seed(0)
#tmp = simulation()
