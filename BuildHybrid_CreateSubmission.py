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
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender

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
SLIMEN_recommender.load_model("Built_Hybrids", SLIMEN_recommender.RECOMMENDER_NAME + "_train_val_test.zip")
gc.collect()

RP3Beta_recommender = RP3betaRecommender(urm_train)
RP3Beta_recommender.load_model("Built_Hybrids", RP3Beta_recommender.RECOMMENDER_NAME + "_train_val_test.zip")
gc.collect()

IALS_recommender = IALSRecommender(urm_train)
IALS_recommender.load_model("Built_Hybrids", IALS_recommender.RECOMMENDER_NAME + "_train_val_test.zip")
gc.collect()

model = GeneralizedLinearHybridRecommender

combo = [SLIMEN_recommender,RP3Beta_recommender,IALS_recommender]

    
#study = op.create_study(direction="maximize")
#study.optimize(objective, n_trials=500)

final = model(urm_train+urm_validation+urm_test, combo, True)
alphas = [1.3782680833859562, 1.8228419650954735,0.13862584225860689]
final.fit(alphas)

df = pd.read_csv("Dataset/data_train.csv")
df.columns = ["UserID","ItemID","Interaction"]

target_users = pd.read_csv("Dataset/data_target_users_test.csv")
target_users.columns = ["UserID"]

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

URM_all = urm_train+urm_validation+urm_test

item_popularity_encoded = np.ediff1d(URM_all.tocsc().indptr)
item_popularity_encoded = np.sort(item_popularity_encoded)

tar_users = target_users["UserID"].astype(int)
topPop_encoded = item_popularity_encoded[-10:]

submission = []

print(np.unique(df["UserID"].values))

for index, user in enumerate(tar_users):
    if (user not in df["UserID"].values):
        item_list_encoded = topPop_encoded
    else:
        item_list_encoded = final.recommend(user2user_encoded[user])[:10]
    item_list = []
    for item_encoded in item_list_encoded:
        item_list.append(item_encoded2item[item_encoded])
    submission.append((user, item_list))


def write_submission(submissions):
    with open("./submission_NOKNN_New_Alphas.csv", "w") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")
            
write_submission(submission)