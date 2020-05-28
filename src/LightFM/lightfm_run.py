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


data_dir = '/scratch/ns4486/recommendations/data/'


epochs = [5, 10, 15]
ranks = [10, 30, 50, 70, 90]
ks = [100, 300, 500]

data_percentage = ['0.01', '0.02', '0.05', '0.1', '0.25', '1']

calculate_rmse = True
calculate_precision_at_k = True

log_file_name = "logs/"+str(time.time()) + ".txt"

for perc in data_percentage:

    train_file = f"interactions_{perc}_train_clean.csv"
    val_file = f"interactions_{perc}_val_clean.csv"

    train_data_path = os.path.join(data_dir, train_file)
    val_data_path = os.path.join(data_dir, val_file)

    df = pd.read_csv(train_data_path, header=None, names=['user_id', 'book_id', 'rating'], dtype={'user_id':'int', 'book_id':'int', 'rating':'int'})
    df_val = pd.read_csv(val_data_path, header=None, names=['user_id', 'book_id', 'rating'], dtype={'user_id':'int', 'book_id':'int', 'rating':'int'})

    train_df, val_df, row_hash, col_hash = adjust_dataframes(df, df_val, 'user_id', 'book_id')
    
    train_coo, val_coo = build_coo_matrices(train_df, val_df, 'user_id', 'book_id', 'rating')
#     train_coo = scipy.sparse.coo_matrix((train_df['rating'].values, (train_df['user_id'].values, train_df['book_id'].values)))

    train_csr, val_csr = build_csr_matrices(train_df, val_df, 'user_id', 'book_id', 'rating')
#     train_csr = scipy.sparse.csr_matrix((train_df['rating'].values, (train_df['user_id'].values, train_df['book_id'].values)))

#     val_csr = scipy.sparse.csr_matrix((val_df['rating'].values, (val_df['user_id'].values, val_df['book_id'].values)), shape = train_csr.shape)
    
    assert train_csr.shape == val_csr.shape

    for epoch in epochs:

        for rank in ranks:

            model = LightFM(no_components=rank, loss='warp', learning_rate=0.05)
            start = time.time()
            model.fit(train_coo, epochs= epoch, num_threads=10)
            time_taken_to_fit = time.time() - start
            total = 0
            
            avg_precision = None
            if calculate_precision_at_k is True:

                for k in ks:

                    _p = precision_at_k(model, test_interactions = val_csr, train_interactions = train_csr, k = k)

                    avg_precision = _p.sum()/len(_p)
                    
                    print(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit},K: {k}, Avg Precision at K{avg_precision}\n")
                    log_file = open(log_file_name, 'a+')
                    log_file.write(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit},K: {k}, Avg Precision at K{avg_precision}\n")
                    log_file.close()
            
            rmse = None
            if calculate_rmse is True:

                scores = model.predict(val_df['user_id'].values, val_df['book_id'].values)
                scores = ((scores - scores.min())/(scores.max() - scores.min())) * 5
                rmse = sqrt(mean_squared_error(val_df['rating'].values, scores))

                print(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit}, RMSE = {rmse}")
                log_file = open(log_file_name, 'a+')
                log_file.write(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit}, RMSE = {rmse}\n")
                log_file.close()
            
            if avg_precision == None and rmse == None:
                print(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit}")
                
                log_file = open(log_file_name, 'a+')
                log_file.write(f"Data Size: {perc}, Epochs: {epoch}, Rank: {rank}, Model Training Time: {time_taken_to_fit}\n")
                log_file.close()
                
            