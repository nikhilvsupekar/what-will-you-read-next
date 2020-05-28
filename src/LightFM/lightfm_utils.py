import time
import os
from math import sqrt

import numpy as np
import pandas as pd
import pickle

import scipy
from sklearn.metrics import mean_squared_error

from lightfm import LightFM
from lightfm.evaluation import precision_at_k

from lightfm_utils import adjust_dataframes, build_coo_matrices, build_csr_matrices

def adjust_dataframes(train_df, val_df, row_name, col_name):
	
	val_df = val_df[val_df[row_name].isin(train_df[row_name])]
	val_df = val_df[val_df[col_name].isin(train_df[col_name])]
	
# 	val_df = pd.merge(val_df, train_df, indicator=True, how='outer').query('_merge=="right_only"').drop('_merge', axis=1)
	
	row_hash = {val: i for i, val in enumerate(set(train_df[row_name].values))}
	col_hash = {val: i for i, val in enumerate(set(train_df[col_name].values))}
	
	def row_hash_function(integer):
		return row_hash[integer]
	
	def col_hash_function(integer):
		return col_hash[integer]
	
	train_df[row_name] = train_df[row_name].apply(row_hash_function)
	train_df[col_name] = train_df[col_name].apply(col_hash_function)
	
	val_df[row_name] = val_df[row_name].apply(row_hash_function)
	val_df[col_name] = val_df[col_name].apply(col_hash_function)
	
	return train_df, val_df, row_hash, col_hash


def build_coo_matrices(train_df, val_df, row_name, col_name, data_name):

	# Assumes that the train_df and val_df are pre-adjusted using the adjust function.

	train_coo = scipy.sparse.coo_matrix((train_df[data_name].values, (train_df[row_name].values, train_df[col_name].values)))
	val_coo = scipy.sparse.coo_matrix((val_df[data_name].values, (val_df[row_name].values, val_df[col_name].values)), shape = train_coo.shape)

	return train_coo, val_coo

def build_csr_matrices(train_df, val_df, row_name, col_name, data_name):

	# Assumes that the train_df and val_df are pre-adjusted using the adjust function.

	train_csr = scipy.sparse.csr_matrix((train_df[data_name].values, (train_df[row_name].values, train_df[col_name].values)))
	val_csr = scipy.sparse.csr_matrix((val_df[data_name].values, (val_df[row_name].values, val_df[col_name].values)), shape = train_csr.shape)

	return train_csr, val_csr