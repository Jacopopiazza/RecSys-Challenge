# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommenders.BaseRecommender import BaseRecommender

class GeneralizedLinearHybridRecommenderColdKNN(BaseRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedLinearHybridRecommender"

    def __init__(self, URM_train, KNN_recommender, recommenders: list, verbose=True, KNN_Under_Interactions=2):
        self.RECOMMENDER_NAME = ''
        self.KNN_Under_Interactions = KNN_Under_Interactions
        self.KNN_recommender = KNN_recommender
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridRecommender'

        super(GeneralizedLinearHybridRecommenderColdKNN, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):
        self.alphas = alphas

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):

        result = self.alphas[0]*self.recommenders[0]._compute_item_score(user_id_array,items_to_compute)
        for index in range(1,len(self.alphas)):
            result = result + self.alphas[index]*self.recommenders[index]._compute_item_score(user_id_array,items_to_compute)
        
        rows_with_less_than_nonzero = []
        for index, user in enumerate(user_id_array):
            
            start_index = self.URM_train.indptr[index]
            end_index = self.URM_train.indptr[index + 1]
            num_nonzero_elements_in_row = end_index - start_index
            if num_nonzero_elements_in_row <= self.KNN_Under_Interactions:
                rows_with_less_than_nonzero.append((index,user))

        for index,user in rows_with_less_than_nonzero:
            result[index] = self.KNN_recommender._compute_item_score(user, items_to_compute)

        

        return result