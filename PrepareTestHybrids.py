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

#Load best params for the models

SLIMEN_best_params = {'topK': 7693, 'l1_ratio': 0.08013148517795793, 'alpha': 0.0012244028139782985}
RP3Beta_best_params = {'topK': 41,
 'alpha': 0.24025759098180052,
 'beta': 0.21463311953617964,
 'normalize_similarity': True}
EASE_best_params = {'topK':None, 'normalize_matrix':False,'l2_norm':84.03422929536671}
ItemKNN_best_params = {'topK': 23, 'shrink': 18, 'similarity': 'tversky', 'normalize': False}
IALS_best_params = {'num_factors': 184, 'epochs': 110, 'confidence_scaling': 'linear', 'alpha': 13.161328184474756, 'epsilon': 0.2917133297273583, 'reg': 0.0005872701636540686}
SLIMBPR_best_params = {'topK': 5, 'epochs': 60, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.1}

#Define and train baseline models

SLIMEN_recommender = SLIMElasticNetRecommender(urm_train+urm_validation)
SLIMEN_recommender.fit(**SLIMEN_best_params)
SLIMEN_recommender.save_model("Hybrids_Experiments", SLIMEN_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()

RP3Beta_recommender = RP3betaRecommender(urm_train+urm_validation)
RP3Beta_recommender.fit(**RP3Beta_best_params)
RP3Beta_recommender.save_model("Hybrids_Experiments", RP3Beta_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()

ITEMKNN_recommender = ItemKNNCFRecommender(urm_train+urm_validation)
ITEMKNN_recommender.fit(**ItemKNN_best_params)
ITEMKNN_recommender.save_model("Hybrids_Experiments", ITEMKNN_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()

IALS_recommender = IALSRecommender(urm_train+urm_validation)
IALS_recommender.fit(**IALS_best_params)
IALS_recommender.save_model("Hybrids_Experiments", IALS_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()

SLIMBPR_recommender = SLIM_BPR_Cython(urm_train+urm_validation)
SLIMBPR_recommender.fit(**SLIMBPR_best_params)
SLIMBPR_recommender.save_model("Hybrids_Experiments", SLIMBPR_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()

EASE_recommender = EASE_R_Recommender(urm_train+urm_validation)
EASE_recommender.fit(**EASE_best_params)
EASE_recommender.save_model("Hybrids_Experiments", EASE_recommender.RECOMMENDER_NAME + "_trainval.zip")
gc.collect()