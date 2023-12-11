
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseSimilarityMatrixRecommender, BaseItemSimilarityMatrixRecommender

import scipy.sparse as sp

class HybridSimilarityMatrixRecommender(BaseItemSimilarityMatrixRecommender):
    """
    This recommender merges N recommendes by weighting their ratings
    """

    RECOMMENDER_NAME = "HybridSimilarityMatrixRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):

        for rec in recommenders:
            assert isinstance(rec, BaseItemSimilarityMatrixRecommender)

        self.RECOMMENDER_NAME = ''
        for recommender in recommenders:
            self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + recommender.RECOMMENDER_NAME[:-11]
        self.RECOMMENDER_NAME = self.RECOMMENDER_NAME + 'HybridSimilarityMatrixRecommender'

        super(HybridSimilarityMatrixRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alphas=None):

        self.W_sparse = sp.csr_matrix((self.recommenders[0].W_sparse.shape[0], self.recommenders[0].W_sparse.shape[1]))

        for index, rec in enumerate(self.recommenders):
            mini = rec.W_sparse.min()
            maxi = rec.W_sparse.max()

            
            if not sp.isspmatrix_csr(rec.W_sparse):
                rec.W_sparse = sp.csr_matrix(rec.W_sparse)


            norma = (rec.W_sparse - mini) / (maxi - mini)

            self.W_sparse += norma * alphas[index]
            self.W_sparse.eliminate_zeros()

        

    def save_model(self, folder_path, file_name=None):
        pass
