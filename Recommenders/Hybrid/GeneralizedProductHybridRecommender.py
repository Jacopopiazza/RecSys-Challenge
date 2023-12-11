# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommenders.BaseRecommender import BaseRecommender

class GeneralizedProductHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedProductHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):
        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridProductRecommender'

        super(GeneralizedProductHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self):
        pass

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        result = self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        for index in range(1,len(self.recommenders)):
            result = result * self.recommenders[index]._compute_item_score(user_id_array,items_to_compute)
        return result

