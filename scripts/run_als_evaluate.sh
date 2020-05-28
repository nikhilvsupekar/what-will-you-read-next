
module load python/gnu/3.6.5
module load spark/2.4.0

. ../properties/spark.properties
. ../properties/app.properties

script_name="als_evaluate"
current_time=$(date +'%Y%m%d_%H%M%S')

log_file_name="$script_name""_""$current_time"".log"
err_file_name="$script_name""_""$current_time"".err"

set -x

spark-submit \
    --name $APP_NAME \
    --master $MASTER \
    --deploy-mode $DEPLOY_MODE \
    --num-executors $NUM_EXECUTORS \
    --executor-memory $EXECUTOR_MEMORY \
    --executor-cores $CORES_PER_EXECUTOR \
    --driver-cores $DRIVER_CORES \
    ../src/"$script_name".py \
        --app-directory $APP_DIRECTORY \
        --sample-percent 0.01 \
        --ranks "$GRID_SEARCH_RANKS" \
        --max-iters "$GRID_SEARCH_MAX_ITERS" \
        --reg-params "$GRID_SEARCH_REG_PARAMS" > ../logs/$log_file_name 2> ../logs/$err_file_name

set +x
