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
import os




#Load urm splitted for repoducibility
urm_train = sps.load_npz("Dataset/urm_train.npz")
urm_train.eliminate_zeros()

urm_test = sps.load_npz("Dataset/urm_test.npz")
urm_test.eliminate_zeros()

urm_validation = sps.load_npz("Dataset/urm_validation.npz")
urm_validation.eliminate_zeros()

#Load best params for the models

SLIMEN_best_params = {'topK': 7693, 'l1_ratio': 0.08013148517795793, 'alpha': 0.0012244028139782985}
RP3Beta_best_params = {'topK': 41, 'alpha': 0.24025759098180052, 'beta': 0.21463311953617964, 'normalize_similarity': True}
EASE_best_params = {'topK':None, 'normalize_matrix':False,'l2_norm':84.03422929536671}
ItemKNN_best_params = {'topK': 23, 'shrink': 18, 'similarity': 'tversky', 'normalize': False}
IALS_best_params = {'num_factors': 184, 'epochs': 110, 'confidence_scaling': 'linear', 'alpha': 13.161328184474756, 'epsilon': 0.2917133297273583, 'reg': 0.0005872701636540686}
SLIMBPR_best_params = {'topK': 5, 'epochs': 60, 'symmetric': False, 'sgd_mode': 'adagrad', 'lambda_i': 1e-05, 'lambda_j': 1e-05, 'learning_rate': 0.1}

#Define and train baseline models
ToTrain = [
    {'model': SLIMElasticNetRecommender, 'params': SLIMEN_best_params},
    {'model': RP3betaRecommender, 'params': RP3Beta_best_params},
    {'model': EASE_R_Recommender, 'params': EASE_best_params},
    {'model': ItemKNNCFRecommender, 'params': ItemKNN_best_params},
    {'model': IALSRecommender, 'params': IALS_best_params},
    {'model': SLIM_BPR_Cython, 'params': SLIMBPR_best_params}
]


if not os.path.isfile("STEPS_Training.csv"):
    df = pd.DataFrame(columns=["Model","Version","Trained"])

    for m in ToTrain:
        df = df.append({'Model':m['model'].RECOMMENDER_NAME, 'Version':0,'Trained':'no'},ignore_index=True)
        df = df.append({'Model':m['model'].RECOMMENDER_NAME, 'Version':1,'Trained':'no'},ignore_index=True)
        df = df.append({'Model':m['model'].RECOMMENDER_NAME, 'Version':2,'Trained':'no'},ignore_index=True)


    df.to_csv("STEPS_Training.csv")
    del df

df = pd.read_csv("STEPS_Training.csv")

FOLDER = "Built_Hybrids"

def trainModelAndStore(model, params, version,df):

    condition = (df['Model'] == model.RECOMMENDER_NAME) & (df['Version'] == version)
    filtered_df = df[condition]
    if not filtered_df.empty and filtered_df["Trained"].iloc[0] == "yes": 
        print(f"{model.RECOMMENDER_NAME} was already trained for version: {version}")
        return

    suffix = ""
    if version == 0:
        instance = model(urm_train)
        suffix = "_train"
    elif version == 1:
        instance = model(urm_train+urm_validation)
        suffix = "_train_val"
    else:
        instance = model(urm_train+urm_validation+urm_test)
        suffix = "_train_val_test"
    
    instance.fit(**params)
    instance.save_model(FOLDER, model.RECOMMENDER_NAME + suffix + ".zip")

    
    df.loc[condition,["Trained"]] = ["yes"]
    df.to_csv("STEPS_Training.csv",index=False)

    del instance
    gc.collect()


for entry in ToTrain:
    model = entry['model']
    params = entry['params']

    print(f"Training {model.RECOMMENDER_NAME} on train set")
    trainModelAndStore(model,params,0,df)
    print(f"Finished training {model.RECOMMENDER_NAME} on train set")
    print(f"Training {model.RECOMMENDER_NAME} on train+val set")
    trainModelAndStore(model,params,1,df)
    print(f"Finished training {model.RECOMMENDER_NAME} on train+val set")
    print(f"Training {model.RECOMMENDER_NAME} on train+val+test set")
    trainModelAndStore(model,params,2,df)
    print(f"Finished training {model.RECOMMENDER_NAME} on train+val+test set")


