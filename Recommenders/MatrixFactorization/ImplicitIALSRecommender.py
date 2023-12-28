import numpy as np


from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix
import implicit



class ImplicitIALSRecommender(BaseRecommender):


    RECOMMENDER_NAME = "Implicit-IALSRecommender"

    def __init__(self, URM_train, factors=50, iterations=25, verbose=True):

        super(BaseRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.model = implicit.als.AlternatingLeastSquares(factors=factors, iterations=iterations)
        self.epochs_done = 0
        self.best_validation_metric = None

    def _run_epoch(self, num_epoch):
        self.model.fit(self.URM_train)

    def _compute_item_score(self, user_id_array, items_to_compute = None):
        _, scores = self.model.recommend(user_id_array, self.URM_train[user_id_array], N=self.URM_train.shape[1])
        return scores

    
    #def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None, remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
    #    
        
    #    print(f"PARAMS: user_id_array={user_id_array}, remove_seen_flag={remove_seen_flag}, items_to_compute={items_to_compute}, remove_top_pop_flag={remove_top_pop_flag}, remove_custom_items_flag={remove_custom_items_flag}, return_scores={return_scores}")
    #    if remove_seen_flag:
    #        self._remove_seen_on_scores()

    #    toBeReturned, scores = self.model.recommend(user_id_array, self.URM_train[user_id_array], N=self.URM_train.shape[1]-self.URM_train[user_id_array].nnz)
    #    if cutoff:
    #        toBeReturned = toBeReturned[:cutoff]
    #    if return_scores:
    #        return toBeReturned.tolist(), scores
    #    return toBeReturned.tolist()

    def _prepare_model_for_validation(self):
        pass

       
    def fit(self,epochs=100, **earlystopping_kwargs):

        if "validation_every_n" in earlystopping_kwargs:
            validation_every_n = earlystopping_kwargs["validation_every_n"]
        if "evaluator_object" in earlystopping_kwargs:
            evaluator_object = earlystopping_kwargs["evaluator_object"]
        if "validation_metric" in earlystopping_kwargs:
            validation_metric = earlystopping_kwargs["validation_metric"]
        if "stop_on_validation" in earlystopping_kwargs:
            stop_on_validation = earlystopping_kwargs["stop_on_validation"]
        if "lower_validations_allowed" in earlystopping_kwargs:
            lower_validations_allowed = earlystopping_kwargs["lower_validations_allowed"]
        if "epochs_min" in earlystopping_kwargs:
            epochs_min = earlystopping_kwargs["epochs_min"]
        else:
            epochs_min = 0

        lower_validatons_count = 0

        for i in range(epochs):
            self._run_epoch(self.epochs_done+1)
            self.epochs_done += 1

            if not validation_every_n is None:
                if self.epochs_done % validation_every_n == 0:
                    results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                    current_metric_value = results_run.iloc[0][validation_metric]

                    print(f"Current MAP: {current_metric_value} @ epoch : {self.epochs_done}")
                    print(f"Curr scores:{ self._compute_item_score([0])}")

                    if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                        print(f"New best model found! MAP: {current_metric_value} @ epoch : {self.epochs_done}")
                        self.best_validation_metric = current_metric_value
                        #self._update_best_model()

                        self.epochs_best = self.epochs_done
                        lower_validatons_count = 0

                    else:
                        lower_validatons_count += 1

                    if stop_on_validation and lower_validatons_count >= lower_validations_allowed and self.epochs_done >= epochs_min:
                        print("{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}".format(
                            self.RECOMMENDER_NAME, self.epochs_done, validation_metric, self.epochs_best, self.best_validation_metric))
                        return