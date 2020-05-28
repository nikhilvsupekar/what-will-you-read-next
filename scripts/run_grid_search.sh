module load python/gnu/3.6.5
module load spark/2.4.0

. ../properties/spark.properties
. ../properties/app.properties

spark-submit \
    --name $APP_NAME \
    --master $MASTER \
    --deploy-mode $DEPLOY_MODE \
    --num-executors $NUM_EXECUTORS \
    --executor-memory $EXECUTOR_MEMORY \
    --executor-cores $CORES_PER_EXECUTOR \
    --driver-cores $DRIVER_CORES \
    ../src/grid_search.py \
        --app-directory $APP_DIRECTORY \
        --ranks $GRID_SEARCH_RANKS \
        --max-iters $GRID_SEARCH_MAX_ITERS \
        --reg-params $GRID_SEARCH_REG_PARAMS