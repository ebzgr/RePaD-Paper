import numpy as np
import pandas as pd
import data_generator as dg
import partitioner as pt
import utility as utl
import pdb
import estimator as st
from hyperopt import fmin, tpe, hp
from functools import partial
from joblib import Parallel, delayed
from hyperopt.pyll import scope


def partitioner_objective(params ,train_data, validation_data, return_full=False):
    partitioner = pt.partitioner(lamb = params['lamb'], delta=params['delta'], min_size = params['min_size'], smoothing_del = params['smoothing_del'])
    parts, report = partitioner.partition(train_data, validation_data)
    max_row = report.loc[len(report)-1]
    min_size = max_row.min_node_size
    if(return_full):
        return max_row
    
    print(params, len(parts), -max_row.test_adjusted_likelihood )
    return -max_row.test_score

def hyper_param_optimizer(data_gen, buses, periods, seed):
    train_df = data_gen.generate(buses=buses,periods=periods)
    valid_df = data_gen.generate(buses=buses,periods=periods)
    tr_data = utl.get_partitioning_variables(train_df)
    vl_data = utl.get_partitioning_variables(valid_df)
    
    objective = partial(partitioner_objective, train_data = tr_data, validation_data = vl_data)
    space = {
        'lamb': hp.choice('lamb', [1]),
        'delta': hp.loguniform('delta', np.log(0.0001), np.log(0.1)),
        'min_size': hp.choice('min_size', [200]),
        'smoothing_del':hp.loguniform('smoothing_del',  np.log(0.000001), np.log(1)),
        'max_pi': scope.int(hp.quiniform('my_param', 4, 100, q=1))
        }
    
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=25, rstate= np.random.RandomState(seed))

    return best

def simulation_1_single_run(data_gen, partitioner, estimator, buses, periods, dim_q):
    train_df = data_gen.generate(buses=buses, periods=periods)
    data = utl.get_partitioning_variables(train_df)
    
    pi = np.zeros(len(train_df))
    tr_bestll,tr_f,tr_alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)

    
    parts, report = partitioner.partition(data)
    pi = utl.q_to_pi_states(parts, data['Q'], dim_q)
    bestll,f,alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)

    pi = np.zeros(len(pi))
    tr_bestll,tr_f,tr_alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)
    
    f.sort()
    return [tr_bestll,tr_alpha,tr_f, bestll, alpha, f, len(parts)]

def monte_carlo(data_gen, estimator, buses, periods, dim_q, N, seed, name='res'):
    best = hyper_param_optimizer(data_gen = data_gen, buses = buses, periods = periods, seed = seed)
    print(best)
    partitioner = pt.partitioner(lamb = 1, 
                                 delta = best['delta'],
                                 min_size = 200, 
                                 smoothing_del = best['smoothing_del'])
    
    df = pd.DataFrame(columns=['tr_ll','tr_alpha','tr_f','ll','alpha','f', 'pi_len'])  

    element_run = Parallel(n_jobs=-1)(delayed(simulation_1_single_run)(data_gen, partitioner, estimator, buses, periods, dim_q) for k in range(N))
    for i in range(N):
        df.loc[len(df)]=element_run[i]
        
    df.to_pickle('data/{}.pickle'.format(name)) 
    return best

def simulation(seed=123):
    dim_q = 10
    max_q = 10
    buses = 100
    periods = 100
    N = 100
    np.random.seed(seed)
    # Generating the data given a partitioning
    partitions = pd.DataFrame(columns = ['state','q_0_min','q_0_max','q_1_min','q_1_max','c'])
    partitions.loc[len(partitions)] = [0,0,5,0,5,-5]
    partitions.loc[len(partitions)] = [1,0,5,5,10,-5]
    partitions.loc[len(partitions)] = [2,5,10,5,10,-5]
    partitions.loc[len(partitions)] = [3,5,10,0,5,-5]
    for i in range(2,dim_q):
        partitions['q_{}_min'.format(i)] = 0
        partitions['q_{}_max'.format(i)] = 10

    hypers = None
    estimator = st.BusEngineNFXP()
    for diff_rep_cost in [True,False]:
        for ml_tr_mode in [True, False]:
            for q_trans_mod in [1,2,3]:
                if(diff_rep_cost):
                    partitions['f_dc'] = np.arange(4,8)*-1
                else:
                    partitions['f_dc'] = np.ones(4)*-5
                    
                if(ml_tr_mode):
                    partitions['f_tr'] = np.arange(1,5)
                else:
                    partitions['f_tr'] = np.ones(4)       
                    
                data_gen = dg.BusEngineDataGenerator(max_mileage=20, mileage_coefficient=-0.2, 
                                                    partitions=partitions, q_transition=q_trans_mod, 
                                                    max_q=10, dim_q=dim_q, discounting_factor=0.9)
                
                res = monte_carlo(data_gen, estimator, buses = 100, periods = 100, dim_q = dim_q, N = 100, seed = seed, name="Monte_mode_{}_diffx_{}_diffcost_{}".format(q_trans_mod, ml_tr_mode*1, diff_rep_cost*1))


#13:31
#simulation_1(seed=123)   
tmp = test(seed = 123)
