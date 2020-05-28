module load python/gnu/3.6.5
module load spark/2.4.0

. ../properties/spark.properties
. ../properties/app.properties

hdfs dfs -rm -r $APP_DIRECTORY/data/*

spark-submit \
    --name $APP_NAME \
    --master $MASTER \
    --num-executors $NUM_EXECUTORS \
    --executor-memory $EXECUTOR_MEMORY \
    ../src/preprocess.py \
        --interactions-file $INTERACTION_FILE \
        --users-file $USERS_FILE\
        --books-file $BOOKS_FILE \
        --output-directory $OUTPUT_DIRECTORY \
        --train-split-percent $TRAIN_SPLIT_PERCENT \
        --validation-split-percent $VALIDATION_SPLIT_PERCENT \
        --interactions-validation-train-percent $INTERACTIONS_VALIDATION_TRAIN_PERCENT \
        --sample-percent $SAMPLE_PERCENT \
        --random-seed $RANDOM_SEED