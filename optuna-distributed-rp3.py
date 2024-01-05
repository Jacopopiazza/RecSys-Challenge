# %%
import os
#os.chdir( "../")

# %%
import pandas as pd
import numpy as np

path = "Dataset/data_train.csv"
df = pd.read_csv(filepath_or_buffer=path,
                               sep=",",
                               header=1,
                               engine='python',
                               names=['UserID', 'ItemID', 'Interaction'])

df

# %%
df.Interaction.value_counts()

# %%
df.info()

# %%
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

# %%
df.head()

# %%
userId_unique = df["UserID"].unique()
itemId_unique = df["ItemID"].unique()

# %%
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample



urm_all = sps.coo_matrix((df["Interaction"].values, 
                          (df["User"].values, df["Item"].values)))

seeds = [1811854, 3772146, 6407100, 6926227, 1340378, 68679, 3822303, 4058970]

urm_train_validation = []
urm_test = []
urm_train = []
urm_validation = []

for seed in seeds:
    a, b = split_train_in_two_percentage_global_sample(urm_all, train_percentage = 0.80,seed=seed)
    c, d = split_train_in_two_percentage_global_sample(a, train_percentage = 0.80, seed=seed)
    urm_train_validation.append(a)
    urm_test.append(b)
    urm_train.append(c)
    urm_validation.append(d)

# %%
num_users = len(userId_unique)
num_items = len(itemId_unique)

evaluator_validation = []
evaluator_test = []
# %%
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Recommender_import_list import *
from Evaluation.Evaluator import EvaluatorHoldout
for i, _ in enumerate(seeds):   
    evaluator_validation.append(EvaluatorHoldout(urm_validation[i], cutoff_list=[10], ignore_users=[]))
    evaluator_test.append(EvaluatorHoldout(urm_test[i], cutoff_list=[10], ignore_users=[]))

# %% [markdown]
# ## Insert model here

# %%
model = RP3betaRecommender

# %%
import optuna as op

def objective(trial):
    topK = trial.suggest_int("topK", 5, 1000)
    #l1 = trial.suggest_float("l1_ratio", 0.01, 0.1)
    alpha = trial.suggest_float("alpha", 0, 1) # maybe [0.5-1]?
    beta = trial.suggest_float("beta", 0, 1)
    #normalize_similarity = trial.suggest_categorical()

    score = 0
    for i, _ in enumerate(seeds):
        recommender = model(urm_train_validation[i])
        recommender.fit(**trial.params, **{"normalize_similarity":True})
        
        result, _ = evaluator_test[i].evaluateRecommender(recommender)
        MAP_result = result["MAP"].item()

        # Assuming your metric of interest is stored in 'metric_value'
        # Replace 'metric_value' with your actual evaluation metric
        if MAP_result < 0.0445:  # Set your threshold for pruning here
            # Prune the trial if the condition is met
            #trial.report(MAP_result, step=i)  # Report the metric
            #if trial.should_prune():  # Check if the trial should be pruned
            print("Pruning trial")
            raise optuna.TrialPruned()
            
        score += MAP_result
    
    score = score /len(seeds)
       
    return score

# %%
best_params = {'topK': 46,
 'alpha': 0.8604387480969916,
 'beta': 0.1537736651871471,
 'normalize_similarity': True}

skiopt_params =  {'topK': 81, 'alpha': 0.2751527191245056, 'beta': 0.1824406673369836, 'normalize_similarity': True}

from joblib import parallel_backend
import logging
import sys

import optuna


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "rp3-study-bayes"  # Unique identifier of the study.
#storage = optuna.storages.JournalStorage(
#    optuna.storages.JournalFileStorage("/mnt/c/Users/Japo/optuna_db/journal.log"),  # NFS path for distributed optimization
#)

mysql_url = "mysql+pymysql://optuna:your_password@192.168.16.83/optuna"
storage = mysql_url

study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction="maximize", sampler=optuna.samplers.CmaEsSampler())

study.enqueue_trial(best_params)

study.enqueue_trial(skiopt_params)

study.optimize(objective, n_trials=200)

# %%
study.trials_dataframe().to_csv("OPTUNA_RP3BETA_V4.csv")

# %%
opt_df = study.trials_dataframe()

# %%
study.best_params

# %%
import matplotlib.pyplot as pyplot

pyplot.scatter(opt_df["params_alpha"].values, opt_df["value"].values, label="OPTUNA")
#pyplot.plot(MAP_per_shrinkage_rnd["shrink"].values, MAP_per_shrinkage_rnd["MAP"].values, label="Rnd")
pyplot.ylabel('MAP')
pyplot.xlabel('alpha')
pyplot.legend()
pyplot.show()

# %%
import matplotlib.pyplot as pyplot

pyplot.scatter(opt_df["params_topK"].values, opt_df["value"].values, label="OPTUNA")
#pyplot.plot(MAP_per_shrinkage_rnd["shrink"].values, MAP_per_shrinkage_rnd["MAP"].values, label="Rnd")
pyplot.ylabel('MAP')
pyplot.xlabel('topK')
pyplot.legend()
pyplot.show()

# %%
import matplotlib.pyplot as pyplot

pyplot.scatter(opt_df["params_normalize_similarity"].values, opt_df["value"].values, label="OPTUNA")
#pyplot.plot(MAP_per_shrinkage_rnd["shrink"].values, MAP_per_shrinkage_rnd["MAP"].values, label="Rnd")
pyplot.ylabel('MAP')
pyplot.xlabel('topK')
pyplot.legend()
pyplot.show()



# %%
import matplotlib.pyplot as pyplot

pyplot.scatter(opt_df["params_beta"].values, opt_df["value"].values, label="OPTUNA")
#pyplot.plot(MAP_per_shrinkage_rnd["shrink"].values, MAP_per_shrinkage_rnd["MAP"].values, label="Rnd")
pyplot.ylabel('MAP')
pyplot.xlabel('beta')
pyplot.legend()
pyplot.show()

# %%
study.best_value

# %%
study.best_params

# %%
final = model(urm_train_validation)
final.fit(**study.best_params)

# %%
from Evaluation.Evaluator import EvaluatorHoldout
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10], ignore_users=[])
evaluator_test.evaluateRecommender(final)


