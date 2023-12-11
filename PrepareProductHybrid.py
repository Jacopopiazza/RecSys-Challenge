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
from Recommenders.Hybrid.HybridSimilarityMatrixRecommender import HybridSimilarityMatrixRecommender
from Recommenders.Hybrid.GeneralizedProductHybridRecommender import GeneralizedProductHybridRecommender



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
SLIMEN_recommender.load_model("Built_Hybrids", SLIMEN_recommender.RECOMMENDER_NAME + "_train.zip")
gc.collect()

RP3Beta_recommender = RP3betaRecommender(urm_train)
RP3Beta_recommender.load_model("Built_Hybrids", RP3Beta_recommender.RECOMMENDER_NAME + "_train.zip")
gc.collect()

IALS_recommender = IALSRecommender(urm_train)
IALS_recommender.load_model("Built_Hybrids", IALS_recommender.RECOMMENDER_NAME + "_train.zip")
gc.collect()

EASE_recommender = EASE_R_Recommender(urm_train)
EASE_recommender.load_model("Built_Hybrids", EASE_recommender.RECOMMENDER_NAME + "_train.zip")
gc.collect()

ITEMKNN = ItemKNNCFRecommender(urm_train)
ITEMKNN.load_model("Built_Hybrids", ITEMKNN.RECOMMENDER_NAME + "_train.zip")
gc.collect()




model = GeneralizedProductHybridRecommender

combo = [SLIMEN_recommender,RP3Beta_recommender,ITEMKNN,EASE_recommender]


recommender = model(urm_train, combo , True)
recommender.fit()

result, _ = evaluator_validation.evaluateRecommender(recommender)

print(result["MAP"])
