#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
#os.chdir( "../")


# In[2]:


import pandas as pd
import numpy as np

path = "Dataset/data_train.csv"
df = pd.read_csv(filepath_or_buffer=path,
                               sep=",",
                               header=1,
                               engine='python',
                               names=['UserID', 'ItemID', 'Interaction'])


df


# In[3]:


user_ids = df["UserID"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
item_ids = df["ItemID"].unique().tolist()
item2item_encoded = {x: i for i, x in enumerate(item_ids)}
item_encoded2item = {i: x for i, x in enumerate(item_ids)}
df["User"] = df["UserID"].map(user2user_encoded)
df["Item"] = df["ItemID"].map(item2item_encoded)

num_users = len(user2user_encoded)
num_items = len(item_encoded2item)
df["Interaction"] = df["Interaction"].values.astype(np.float32)

# min and max ratings will be used to normalize the ratings later
min_rating = 0.0
max_rating = max(df["Interaction"])

print(
    "Number of users: {}, Number of Items: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_items, min_rating, max_rating
    )
)


# In[4]:


userId_unique = df["UserID"].unique()
itemId_unique = df["ItemID"].unique()


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

seeds = [57861821,16707467,65130230,43705915,77606739,55325039,37576322,42227625,86290524,12764724]

urm_all = sps.coo_matrix((df["Interaction"].values, 
                          (df["User"].values, df["Item"].values)))

urm_train_validation = []
urm_test = []
urm_train = []
urm_validation = []

for i in range(len(seeds)):
    a, b = split_train_in_two_percentage_global_sample(urm_all, train_percentage = 0.80,seed=seeds[i])
    c, d = split_train_in_two_percentage_global_sample(a, train_percentage = 0.80,seed=seeds[i])
    urm_train_validation.append(a)
    urm_test.append(b)
    urm_train.append(c)
    urm_validation.append(d)



# In[6]:


num_users = len(userId_unique)
num_items = len(itemId_unique)


# In[7]:


from Recommenders.Recommender_import_list import *
from Evaluation.Evaluator import EvaluatorHoldout

evaluator_validation = []
for u in urm_validation:
    evaluator_validation.append(EvaluatorHoldout(u, cutoff_list=[10], ignore_users=[]))


# In[8]:


from Evaluation.Evaluator import EvaluatorHoldout
evaluator_test = []
for u in urm_test:
    evaluator_test.append(EvaluatorHoldout(u, cutoff_list=[10], ignore_users=[]))


# In[9]:


import os

output_folder_path = "result_experiments_parallel/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
    
n_cases = 20  # using 10 as an example
n_random_starts = int(n_cases*0.3)
metric_to_optimize = "MAP"   
cutoff_to_optimize = 10


# In[10]:


from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content
from Recommenders.NonPersonalizedRecommender import TopPop, Random
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender
import os, multiprocessing
from functools import partial
from skopt.space import Real, Integer, Categorical


# In[11]:


earlystopping_keywargs = {"validation_every_n": 5,
                          "stop_on_validation": True,
                          "evaluator_object": evaluator_validation,
                          "lower_validations_allowed": 5,
                          "validation_metric": metric_to_optimize,
                          }


# In[12]:


from concurrent.futures import ProcessPoolExecutor
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import time


# In[13]:


n_points = 10000
n_jobs = 1  # Using all available CPUs
acq_func = 'gp_hedge'
acq_optimizer = 'auto'
verbose = True
n_restarts_optimizer = 10
xi = 0.01
kappa = 1.96
x0 = None
y0 = None
noise = 1e-5
random_state = int(os.getpid() + time.time()) % np.iinfo(int).max
model_counter=0


# In[14]:


recommenderClass = SLIMElasticNetRecommender

hyperparameters_range_dictionary = {
        "topK": Integer(5, 300),
        "alpha": Real(low = 0.2651626829923486, high = 0.37644615066224263, prior = 'uniform'),
        "beta": Real(low = 0, high = 1.35, prior = 'uniform'),
        "normalize_similarity": Categorical([True, False]),
        #"implicit" : Categorical([True])
}

hyperparameters_range_dictionary = {
                "topK": Integer(5, 1000),
                "l1_ratio": Real(low = 1e-5, high = 1.0, prior = 'log-uniform'),
                "alpha": Real(low = 1e-3, high = 1.0, prior = 'uniform'),
            }


# In[15]:


from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a logger for the run_fold function
logger = logging.getLogger("run_fold")

# Define the number of processes you want to use
num_processes = multiprocessing.cpu_count()


print(f"RUNNING at most {num_processes} processes")

sum_map = lambda results: sum(item[metric_to_optimize] for item in results)

hyperparams = dict()
hyperparams_names = list()
hyperparams_values = list()

skopt_types = [Real, Integer, Categorical]

for name, hyperparam in hyperparameters_range_dictionary.items():

    if any(isinstance(hyperparam, sko_type) for sko_type in skopt_types):
        hyperparams_names.append(name)
        hyperparams_values.append(hyperparam)
        hyperparams[name] = hyperparam

columns = ["Fold",metric_to_optimize]
for h in hyperparams_names:
    columns.append(h)

fold_results_df = pd.DataFrame(columns=columns)

def evaluate_model(hyperparams):

    logger.info("start")

    global fold_results_df

    current_fit_hyperparameters_dict = dict(zip(hyperparams_names, hyperparams))

    def run_fold(fold, **hyperparams):
        # Fit the model on this fold and return the evaluation metric
        # ...
        logger.info(f"Fold {fold}: starting to fit...")

        recommender = recommenderClass(urm_train[fold], verbose=True)
        recommender.fit(**hyperparams)
        result, _ = evaluator_validation[fold].evaluateRecommender(recommender)
        metric_result = result[metric_to_optimize].item()
        
        logger.info(f"Fold {fold}: hyperparams {hyperparams}, ended with {metric_to_optimize} {metric_result}")

        return {'Fold': fold, metric_to_optimize: metric_result, **hyperparams}

    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        process_futures = [executor.submit(lambda fold: run_fold(fold, **current_fit_hyperparameters_dict), fold)
                           for fold in range(len(seeds))]
    
     # Wait for all processes to complete
    fold_results = []
    for future in as_completed(process_futures):
        fold_results.append(future.result())
    
    logger.info(fold_results)
    for res in fold_results:
        fold_results_df = fold_results_df.append(res, ignore_index=True)
    fold_results_df.to_csv(output_folder_path + recommenderClass.RECOMMENDER_NAME + "_folds_results.csv")

    # Compute average score across all folds
    average_score = sum_map(fold_results) / len(fold_results)

    logger.info(f"Finished trial with Average {metric_to_optimize}: {average_score}")

    return -average_score  # Assuming a score that needs to be minimized


# In[16]:


result = gp_minimize(evaluate_model, hyperparams_values, n_calls=100,
                     base_estimator=None,
                     n_initial_points= max(0, n_random_starts - model_counter),
                     initial_point_generator = "random",
                     acq_func=acq_func,
                     acq_optimizer=acq_optimizer,
                     x0=x0,
                     y0=y0,
                     random_state=random_state,
                     verbose=verbose,
                     callback=None,
                     n_points=n_points,
                     n_restarts_optimizer=n_restarts_optimizer,
                     xi=xi,
                     kappa=kappa,
                     noise=noise,
                     n_jobs=1
                    )


# In[ ]:


import pandas as pd

results_df = pd.DataFrame(result.x_iters, columns=[dim for dim in hyperparameters_range_dictionary])
results_df['score'] = result.func_vals


# In[ ]:


results_df['score'].min()


# In[ ]:


SLIMEN = {'topK': 530, 'l1_ratio': 0.05017569359096808, 'alpha': 0.001}
P3ALPHA = {'topK': 400, 'alpha': 1.6632815179401539, 'normalize_similarity': True}
RP3 = {'topK': 71, 'alpha': 0.31274648571776065, 'beta': 0.3586324430664178, 'normalize_similarity': True}

SLIMEN_KFOLD = {'Fold': 0, 'MAP': 0.029228665181843458, 'topK': 660, 'l1_ratio': 0.0012296801858497721, 'alpha': 0.001}
RP3_KFOLD = {'topK': 59,
 'alpha': 0.3764461506622426,
 'beta': 0.10277510174112,
 'normalize_similarity': True}

