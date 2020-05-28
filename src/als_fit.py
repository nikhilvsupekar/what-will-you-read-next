import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
import time
import parser

def string_to_list(x):
    x = x.replace(' ', '').split(',')
    return [float(y) for y in x]


parser = parser.build_parser()
(options, args) = parser.parse_args()

APP_DIRECTORY = options.app_directory
SAMPLE_PERCENT = options.sample_percent
DATA_DIRECTORY = os.path.join(APP_DIRECTORY, 'data')
MODELS_DIRECTORY = os.path.join(APP_DIRECTORY, 'models')

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

print('loading data')
interactions_train = spark.read \
    .format('csv') \
    .schema(interaction_schema) \
    .load(os.path.join(DATA_DIRECTORY, f'interactions_{SAMPLE_PERCENT}_train.csv'))

print('data loaded')

als = ALS(
    userCol = 'user_id',
    itemCol = 'book_id',
    ratingCol = 'rating',
    coldStartStrategy = 'drop'
)

for rank in RANKS:
    for maxIter in MAX_ITERS:
        for regParam in REG_PARAMS:
            print(f'training config: SAMPLE_PERCENT = {SAMPLE_PERCENT}, RANK = {rank}, ITERS = {maxIter}, LAMBDA = {regParam}')
            rank = int(rank)
            maxIter = int(maxIter)
            
            print("Running for " + str((rank, maxIter, regParam)))
            als.setParams(rank = rank, maxIter = maxIter, regParam = regParam)
            
            start = time.time()
            model = als.fit(interactions_train)
            model.save(
                os.path.join(
                    MODELS_DIRECTORY, 'als' + '_s' + str(SAMPLE_PERCENT) + '_r' + str(rank) + '_i' + str(maxIter) + '_l' + str(regParam) + '.model'
                )
            )
            end = time.time()
            run_time = end - start
            
            print('fit time = ' + str(run_time))
            print(f'PARSE|{SAMPLE_PERCENT},{rank},{maxIter},{regParam},{run_time}')