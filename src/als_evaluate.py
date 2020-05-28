import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import *
import parser
import time

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
            
            print(f'evaluating config: SAMPLE_PERCENT = {SAMPLE_PERCENT}, RANK = {rank}, ITERS = {maxIter}, LAMBDA = {regParam}')
            
            start = time.time()
            model = ALSModel.load(
                os.path.join(
                    MODELS_DIRECTORY, 'als' + '_s' + str(SAMPLE_PERCENT) + '_r' + str(rank) + '_i' + str(maxIter) + '_l' + str(regParam) + '.model'
                )
            )
            
            predictions = model.transform(interactions_val)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            end = time.time()
            print("Root-mean-square error = " + str(rmse))
            
            run_time = end - start
            
            print('evaluate time = ' + str(run_time))
            print(f'PARSE|{SAMPLE_PERCENT},{rank},{maxIter},{regParam},{rmse},{run_time}')
