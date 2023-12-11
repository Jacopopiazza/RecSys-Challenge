from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import gc
from Recommenders.Recommender_import_list import *
from Recommenders.EASE_R.EASE_R_RecommenderV2 import EASE_R_RecommenderV2
from Evaluation.Evaluator import EvaluatorHoldout
import optuna as op
from itertools import combinations
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender, GeneralizedNormalizedLinearHybridRecommender

#Load urm splitted for repoducibility
urm_train = sps.load_npz("Dataset/urm_train.npz")
urm_train.eliminate_zeros()

urm_test = sps.load_npz("Dataset/urm_test.npz")
urm_test.eliminate_zeros()

urm_validation = sps.load_npz("Dataset/urm_validation.npz")
urm_validation.eliminate_zeros()

#build evaluators

evaluator_validation = EvaluatorHoldout(urm_validation, cutoff_list=[10], ignore_users=[])
evaluator_test = EvaluatorHoldout(urm_test, cutoff_list=[10], ignore_users=[])

#Define and train baseline models

SLIMEN_recommender = SLIMElasticNetRecommender(urm_train)
SLIMEN_recommender.load_model("Built_Hybrids", SLIMEN_recommender.RECOMMENDER_NAME + "_train_val.zip")
gc.collect()

RP3Beta_recommender = RP3betaRecommender(urm_train)
RP3Beta_recommender.load_model("Built_Hybrids", RP3Beta_recommender.RECOMMENDER_NAME + "_train_val.zip")
gc.collect()

IALS_recommender = IALSRecommender(urm_train)
IALS_recommender.load_model("Built_Hybrids", IALS_recommender.RECOMMENDER_NAME + "_train_val.zip")
gc.collect()

model = GeneralizedLinearHybridRecommender

combo = [SLIMEN_recommender,RP3Beta_recommender,IALS_recommender]

#Optuna objective function
def objective(trial):

    #min_interaction = trial.suggest_int("min_interactions",1,25)
    alphas = [trial.suggest_float(f"alpha_{i+1}", 0.01, 1) for i in range(len(combo))]

    #recommender = model(urm_train, ITEMKNN_recommender, combo,True,min_interaction)
    recommender = model(urm_train, combo , True)
    recommender.fit(alphas = alphas)
    
    result, _ = evaluator_validation.evaluateRecommender(recommender)
    MAP_result = result["MAP"].item()

    gc.collect()
    
    return MAP_result
    
#study = op.create_study(direction="maximize")
#study.optimize(objective, n_trials=500)

mod = model(urm_train+urm_validation, combo, True)
alphas = [1.3782680833859562, 1.8228419650954735,0.13862584225860689]
mod.fit(alphas)

res, _ = evaluator_test.evaluateRecommender(mod)

print(res["MAP"])



