import numpy as np
import pandas as pd
import data_generator as dg
import partitioner as pt
import utility as utl
import pdb
import time
import estimator as st

import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning,ConvergenceWarning
warnings.simplefilter('ignore', HessianInversionWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

def check_divergence():
    dim_q = 10
    buses=500
    periods=100
    q_transition=2
    q_based_mileage_transition=True
    train_df = pd.read_pickle("data/diverge_sample_2.pickle")
    p = pt.partitioner(lamb = 1, delta=0.002, min_size = 1500, smoothing_del = 0.00001)
    data = get_partitioning_variables(train_df)
    parts, report = p.partition(data)
    estimator = st.BusEngineNFXP()
    pi = utl.q_to_pi_states(parts, data['Q'], dim_q)
    bestll,f,alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)
    
    return f,alpha


def get_partitioning_variables(df):
    df=df.rename(columns={'m':'x0'})
    ids = df.id.values
    periods = df.t.values
    X = df[['x0']].values
    Q = df[df.columns[df.columns.str.contains('q_')]].values
    Y = df.d.values
    return {'ids':ids, 'periods':periods, 'X':X, 'Q':Q, 'Y':Y}

def conventional_method_comparison(partitions, buses,periods, q_transition=1, q_based_mileage_transition=True):   
    
    data_engine = dg.BusEngineDataGenerator(max_mileage=30, mileage_coefficient=-0.2,q_initial=None, 
                                     q_discretization=partitions,q_transition=q_transition, 
                                     max_q=10, dim_q=dim_q, discounting_factor=0.9,
                                     q_based_mileage_transition=q_based_mileage_transition)

    estimator = st.BusEngineNFXP()
    train_df = data_engine.generate(buses=buses,periods=periods)
    data = get_partitioning_variables(train_df)
    train_df.to_pickle("data/diverge_sample.pickle")

    pdb.set_trace()
#    return
#    p = pt.partitioner(lamb = 1, delta=0.0002, min_size = 1500, smoothing_del = 0.00001)
#    summary, full= p.cross_validate_partition(data, 3)

# =============================================================================
#     test_df = data_engine.generate(buses=buses, periods=periods)
#     test_data = get_partitioning_variables(test_df)
# 
# #    parts = pd.read_pickle('data/sample_parts.pickle')
#     parts, report = p.partition(data,test_data)
#     pi = utl.q_to_pi_states(parts, data['Q'], dim_q)
#     train_df['pi'] = pi
#     estimator = st.BusEngineNFXP()
#     bestll,f,alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)
#     print(bestll,f,alpha)
# 
#     if(abs(alpha)>0.25):
#         pdb.set_trace()
# =============================================================================
    pi = np.zeros(len(train_df))
    bestll,f,alpha = estimator.estimate_theta(data['ids'], data['periods'], data['X'].flatten().tolist(), pi, data['Y'], beta = 0.9, eps=10**-3)
    print(bestll,f,alpha)


np.random.seed(43)
dim_q = 10

# Generating the data given a partitioning
partitions = pd.DataFrame(columns = ['state','q_0_min','q_0_max','q_1_min','q_1_max','f_dc'])
partitions.loc[len(partitions)] = [0,0,5,0,5,-5]
partitions.loc[len(partitions)] = [1,0,5,5,10,-5]
partitions.loc[len(partitions)] = [2,5,10,5,10,-5]
partitions.loc[len(partitions)] = [3,5,10,0,5,-5]
for i in range(2,dim_q):
    partitions['q_{}_min'.format(i)] = 0
    partitions['q_{}_max'.format(i)] = 10

partitions['f_dc'] = np.arange(4,8)*-1
c = np.arange(4,8)*-1
for i in range(1000):
    conventional_method_comparison(partitions, buses=100, periods=500, q_transition=2, q_based_mileage_transition=False)
print(check_divergence())