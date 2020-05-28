import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import struct, collect_list
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F



import parser

def string_to_list(x):
    x = x.replace(' ', '').split(',')
    return [float(y) for y in x]

parser = parser.build_parser()
(options, args) = parser.parse_args()

APP_DIRECTORY = options.app_directory
DATA_DIRECTORY = os.path.join(APP_DIRECTORY, 'data')
MODELS_DIRECTORY = os.path.join(APP_DIRECTORY, 'models')
SAMPLE_PERCENT = options.sample_percent

RANKS = string_to_list(options.ranks)
MAX_ITERS = string_to_list(options.max_iters) 
REG_PARAMS = string_to_list(options.reg_params)


spark = SparkSession.builder.getOrCreate()

interaction_schema = StructType(
    [
        StructField('user_id', IntegerType(), True),
        StructField('book_id', IntegerType(), True),
        StructField('rating', IntegerType(), True)
    ]
)

interactions_val = spark.read \
    .format('csv') \
    .schema(interaction_schema) \
    .load(os.path.join(DATA_DIRECTORY, f'interactions_{SAMPLE_PERCENT}_val.csv'))

interactions_val = interactions_val.filter('user_id is not null')

for rank in RANKS:
    for maxIter in MAX_ITERS:
        for regParam in REG_PARAMS:
            
            rank = int(rank)
            maxIter = int(maxIter)
            
            model = ALSModel.load(
                os.path.join(
                    MODELS_DIRECTORY, 'als' + '_s' + str(SAMPLE_PERCENT) + '_r' + str(rank) + '_i' + str(maxIter) + '_l' + str(regParam) + '.model'
                )
            )
            
            start = time.time()
            all_recommendations = model.recommendForAllUsers(500)
            
            interactions_val_grouped = interactions_val \
                .withColumn('structs', struct('book_id', 'rating')) \
                .groupby('user_id') \
                .agg(collect_list('structs').alias('group_list'))
                
            predictedItems = all_recommendations.select('user_id', col('recommendations')['book_id'].alias('books'))
            actualItems = interactions_val_grouped.select('user_id', col('group_list')['book_id'].alias('books'))
            
            pred_and_truth = predictedItems.join(
                actualItems,
                predictedItems.user_id == actualItems.user_id,
                'inner'
            ).select(
                predictedItems.user_id,
                predictedItems.books,
                actualItems.books
            )
            
            perUserItemsRDD = pred_and_truth .rdd.map(lambda row: (row[1], row[2]))
            
            rankingMetrics = RankingMetrics(perUserItemsRDD)
            
            MAP = rankingMetrics.meanAveragePrecision
            precision_100 = rankingMetrics.precisionAt(100)
            precision_300 = rankingMetrics.precisionAt(300)
            precision_500 = rankingMetrics.precisionAt(500)
            
            ndcg_100 = rankingMetrics.ndcgAt(100)
            ndcg_300 = rankingMetrics.ndcgAt(300)
            ndcg_500 = rankingMetrics.ndcgAt(500)
            
            end = time.time()
            run_time = end - start
            
            print(f'PARSE|{SAMPLE_PERCENT},{rank},{maxIter},{regParam},{MAP},{precision_100},{precision_300},{precision_500},{ndcg_100},{ndcg_300},{ndcg_500},{run_time}')
