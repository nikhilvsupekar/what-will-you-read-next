import os
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import parser


def string_to_list(x):
    x = x.replace(' ', '').split(',')
    return [float(y) for y in x]


parser = parser.build_parser()
(options, args) = parser.parse_args()

APP_DIRECTORY = options.app_directory
DATA_DIRECTORY = os.path.join(APP_DIRECTORY, 'data')
MODELS_DIRECTORY = os.path.join(APP_DIRECTORY, 'models')

RANKS = string_to_list(options.ranks)
MAX_ITERS = string_to_list(options.max_iters)
REG_PARAMS = string_to_list(options.reg_params)

spark = SparkSession.builder.getOrCreate()
interactions_train = spark.read \
    .format('csv') \
    .options(header='true', inferSchema = 'true') \
    .load(os.path.join(DATA_DIRECTORY, 'interactions_train_full.csv'))

interactions_val = spark.read \
    .format('csv') \
    .options(header='true', inferSchema = 'true') \
    .load(os.path.join(DATA_DIRECTORY, 'interactions_val.csv'))

als = ALS(
    userCol = 'user_id',
    itemCol = 'book_id',
    ratingCol = 'rating',
    coldStartStrategy = 'drop'
)

for rank in RANKS:
    for maxIter in MAX_ITERS:
        for regParam in REG_PARAMS:

            rank = int(rank)
            maxIter = int(maxIter)

            print("Running for " + str((rank, maxIter, regParam)))
            als.setParams(rank=rank, maxIter=maxIter, regParam=regParam)

            model = als.fit(interactions_train)
            # model.save(os.path.join(MODELS_DIRECTORY, 'als'))

            predictions = model.transform(interactions_val)
            evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print("Root-mean-square error = " + str(rmse))

            f = open("validation_errors.csv", "a+")
            f.write(str(rank) +"," + str(maxIter) + "," + str(regParam) + "," + str(rmse) +"\n")
            f.close()
            print("Finshed running for " + str((rank, maxIter, regParam)))
