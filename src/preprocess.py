import os
import math
from optparse import OptionParser

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F

import parser

parser = parser.build_parser()
(options, args) = parser.parse_args()

INTERACTIONS_FILE_PATH = options.interactions_file_path
USERS_FILE_PATH = options.users_file_path
BOOKS_FILE_PATH = options.books_file_path
OUTPUT_DIRECTORY = options.output_directory
TRAIN_SPLIT_PERCENT = options.train_split_percent
VALIDATION_SPLIT_PERCENT = options.validation_split_percent
TEST_SPLIT_PERCENT = 1 - TRAIN_SPLIT_PERCENT - VALIDATION_SPLIT_PERCENT
RANDOM_SEED = options.random_seed
INTERACTIONS_VALIDATION_TRAIN_PERCENT = options.interactions_validation_train_percent
SAMPLE_PERCENT = options.sample_percent
spark = SparkSession.builder.getOrCreate()


def get_interactions_by_users(interactions, users):
    return interactions.join(
        users,
        users.user_id == interactions.user_id,
        'inner'
    ).select(
        interactions.user_id,
        interactions.book_id,
        interactions.rating
    )

# drops users with less than k interactions
def drop_interactions(interactions, k = 10):
    user_counts = interactions.select('user_id').groupby('user_id').count()
    sparse_users = user_counts.filter(user_counts['count'] > k).select('user_id')
    return interactions.join(
        sparse_users,
        interactions.user_id == sparse_users.user_id,
        'inner'
    ).select(
        interactions.user_id,
        interactions.book_id,
        interactions.rating
    )

# splits interactions equally into train and val per user
def split_user_interactions(interactions):
    interactions_rand = interactions.withColumn(
        'rand_num', F.rand()
    )
    
    interactions_with_ranks = interactions_rand.select(
        'user_id',
        'book_id',
        'rating',
        F.percent_rank().over(
            Window.partitionBy(
                interactions_rand['user_id']
            ).orderBy(
                interactions_rand['rand_num']
            )
        ).alias('book_id_rank')
    )
    
    interactions_train = interactions_with_ranks.filter(
        interactions_with_ranks['book_id_rank'] >= 0.5
    ).select(
        'user_id',
        'book_id',
        'rating'
    )
    
    interactions_val = interactions_with_ranks.filter(
        interactions_with_ranks['book_id_rank'] < 0.5
    ).select(
        'user_id',
        'book_id',
        'rating'
    )
    
    return interactions_train, interactions_val


# read all interactions available
interactions = spark.read \
        .format('csv') \
        .options(header='true', inferSchema = 'true') \
        .load(INTERACTIONS_FILE_PATH)

# filter records with 0 ratings
interactions = interactions.filter(interactions['rating'] > 0).select(
    'user_id',
    'book_id',
    'rating'
)

interactions = interactions.sample(
    withReplacement = False, 
    fraction = SAMPLE_PERCENT, 
    seed = RANDOM_SEED
) 

# find list of users
users = interactions.select('user_id').distinct()

# split users into train, val, test
users_train, users_val, users_test = users.randomSplit(
    [TRAIN_SPLIT_PERCENT, VALIDATION_SPLIT_PERCENT, TEST_SPLIT_PERCENT],
    RANDOM_SEED
)

# get interactions for train, val, test users
interactions_train = get_interactions_by_users(interactions, users_train)
interactions_val = get_interactions_by_users(interactions, users_val)
interactions_test = get_interactions_by_users(interactions, users_test)

# split val and test interactions equally per user into train and val splits
interactions_val_train, interactions_val_val = split_user_interactions(interactions_val)
interactions_test_train, interactions_test_test = split_user_interactions(interactions_test)

# drop users with less than 10 interactions per set
interactions_train = drop_interactions(interactions_train, 10)
interactions_val_train = drop_interactions(interactions_val_train, 10)
interactions_val_val = drop_interactions(interactions_val_val, 10)
interactions_test_train = drop_interactions(interactions_test_train, 10)
interactions_test_test = drop_interactions(interactions_test_test, 10)

# create unified train, val, test interaction sets
interactions_train = interactions_train.union(interactions_val_train).union(interactions_test_train)

# keep items observed only during training
items_train = interactions_train.select('book_id').distinct()

interactions_val_val = interactions_val_val.join(
    items_train,
    interactions_val_val['book_id'] == items_train['book_id'],
    'inner'
).select(
    interactions_val_val['user_id'],
    interactions_val_val['book_id'],
    interactions_val_val['rating']
)

interactions_test_test = interactions_test_test.join(
    items_train,
    interactions_test_test['book_id'] == items_train['book_id'],
    'inner'
).select(
    interactions_test_test['user_id'],
    interactions_test_test['book_id'],
    interactions_test_test['rating']
)

save_file_name = 'interactions_' + str(SAMPLE_PERCENT)

interactions_train.write.csv(os.path.join(OUTPUT_DIRECTORY, save_file_name + '_train.csv'), header = True)
interactions_val_val.write.csv(os.path.join(OUTPUT_DIRECTORY, save_file_name + '_val.csv'), header = True)
interactions_test_test.write.csv(os.path.join(OUTPUT_DIRECTORY, save_file_name + '_test.csv'), header = True)
