#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/12/2023

@author: JP
"""

import pandas as pd
from Data_manager.DataReader import DataReader
from Data_manager.DatasetMapperManager import DatasetMapperManager
import scipy.sparse as sps
import numpy as np

class BooksReader(DataReader):

    DATASET_SUBFOLDER = "~/RecSys_Course_AT_PoliMi/Dataset/"
    AVAILABLE_URM = ["URM_all"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []

    IS_IMPLICIT = True

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER


    def _load_from_original_file(self):

        self._print("Loading Interactions")

        URM_all_dataframe = pd.read_csv(self.DATASET_SUBFOLDER + "data_train.csv", 
                       sep=",", 
                       skiprows=[0],
                       names=["UserID", "ItemID", "Data"],
                       header=None,
                       dtype={"UserID": str,
                               "ItemID": str,
                               "Data": np.float32,
                             })

        
        

        # min and max ratings will be used to normalize the ratings later
        min_rating = 0.0
        max_rating = max(URM_all_dataframe["Data"])


        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)

        self._print("Loading Complete")

        return loaded_dataset

