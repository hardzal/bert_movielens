import bz2
import csv
import json
import logging
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_preprocessing.BaseDataset import BaseDataset

class ANIMEDataset(BaseDataset):
    def __init__(self, input_path, output_path):
        super(ANIMEDataset, self).__init__(input_path, output_path)
        self.dataset_name = 'anime'

        # input file
        self.inter_file = os.path.join(self.input_path, 'rating.csv')
        self.item_file = os.path.join(self.input_path, 'anime.csv')

        self.sep = ','

        # output file
        output_files = self.get_output_files()
        self.output_inter_file = output_files[0]
        self.output_item_file = output_files[1]

        # selected feature fields
        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'rating:float'}
        self.item_fields = {0: 'item_id:token',
                            1: 'name:token_seq',
                            2: 'genre:token_seq',
                            3: 'type:token',
                            4: 'episodes:float',
                            5: 'avg_rating:float',
                            6: 'members:float'}

    def load_inter_data(self):
        return pd.read_csv(self.inter_file, delimiter=self.sep, header=None, engine='python').iloc[1:, :]

    def load_item_data(self):
        origin_data = pd.read_csv(self.item_file, delimiter=self.sep, header=None, engine='python').iloc[1:, :]
        processed_data = origin_data
        for i in range(origin_data.shape[0]):
            try:
                split_type = origin_data.iloc[i, 2].split(', ')
                type_str = ' '.join(split_type)
            except:
                type_str = ''
            processed_data.iloc[i, 2] = type_str
        processed_data = processed_data.where((processed_data.applymap(lambda x: True if str(x) != 'nan' else False)),
                                              '')
        processed_data = processed_data.where(
            (processed_data.applymap(lambda x: True if str(x) != 'Unknown' else False)),
            '')
        return processed_data

