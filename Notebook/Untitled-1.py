


tar_users = target_users["UserID"].astype(int)
topPop_encoded = item_popularity_encoded[-10:]

submission = []

print(np.unique(df["UserID"].values))

for index, user in enumerate(tar_users):
    if (user not in df["UserID"].values):
        item_list_encoded = topPop_encoded
    else:
        item_list_encoded = SLIMEN_final_recommender.recommend(user2user_encoded[user])[:10]
    item_list = []
    for item_encoded in item_list_encoded:
        item_list.append(item_encoded2item[item_encoded])
    submission.append((user, item_list))




from Recommenders.BaseRecommender import BaseRecommender

class ScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        
        
    def fit(self, alpha = 0.5):
        self.alpha = alpha      


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1*self.alpha + item_weights_2*(1-self.alpha)

        return item_weights