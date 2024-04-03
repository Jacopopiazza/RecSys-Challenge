import numpy as np


from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
import implicit



class ImplicitIALSRecommenderWrap(BaseRecommender):


    RECOMMENDER_NAME = "Implicit-IALSRecommender"

    def __init__(self, model,urm_train):

        super(BaseRecommender, self).__init__()
        self.model = model
        self.URM_train = urm_train


    def _compute_item_score(self, user_id_array, items_to_compute = None):
        _, scores = self.model.recommend(user_id_array, self.URM_train[user_id_array], N=self.URM_train.shape[1])
        return scores
