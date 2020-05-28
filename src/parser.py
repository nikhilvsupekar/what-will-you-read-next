from optparse import OptionParser


def build_parser():
    parser = OptionParser()
    parser.add_option(
        '-a', 
        '--app-directory',
        dest = 'app_directory',
        default = 'hdfs:/user/ns4486/recommendations',
        type = 'string',
        help = 'HDFS directory for app resources'
    )

    parser.add_option(
        '-i', 
        '--interactions-file',
        dest = 'interactions_file_path',
        default = 'hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv',
        type = 'string',
        help = 'HDFS file path to interactions file'
    )

    parser.add_option(
        '-u', 
        '--users-file',
        dest = 'users_file_path',
        default = 'hdfs:/user/bm106/pub/goodreads/user_id_map.csv',
        type = 'string',
        help = 'HDFS file path to users file'
    )

    parser.add_option(
        '-b', 
        '--books-file',
        dest = 'books_file_path',
        default = 'hdfs:/user/bm106/pub/goodreads/book_id_map.csv',
        type = 'string',
        help = 'HDFS file path to books file'
    )

    parser.add_option(
        '-o', 
        '--output-directory',
        dest = 'output_directory',
        default = 'hdfs:/user/bm106/pub/goodreads/book_id_map.csv',
        type = 'string',
        help = 'output HDFS directory'
    )

    parser.add_option( 
        '--sample-percent',
        dest = 'sample_percent',
        default = '1',
        type = 'float',
        help = 'sampling percentage for interactions data'
    )

    parser.add_option(
        '--train-split-percent',
        dest = 'train_split_percent',
        default = 0.6,
        type = 'float',
        help = 'train split percentage'
    )

    parser.add_option(
        '--validation-split-percent',
        dest = 'validation_split_percent',
        default = 0.2,
        type = 'float',
        help = 'validation split percentage'
    )

    parser.add_option(
        '--interactions-validation-train-percent',
        dest = 'interactions_validation_train_percent',
        default = 0.5,
        type = 'float',
        help = 'percentage of records from interactions validation set to be merged into train set'
    )

    parser.add_option(
        '-s', 
        '--random-seed',
        dest = 'random_seed',
        default = 666,
        type = 'int',
        help = 'random seed'
    )

    parser.add_option(
        '--ranks',
        dest = 'ranks',
        default = '10',
        type = 'string',
        help = 'comma separated list of rank hyperparameter values'
    )

    parser.add_option(
        '--max-iters',
        dest = 'max_iters',
        default = '15',
        type = 'string',
        help = 'comma separated list of max_iters hyperparameter values'
    )

    parser.add_option(
        '--reg-params',
        dest = 'reg_params',
        default = '0.1',
        type = 'string',
        help = 'comma separated list of reg_params hyperparameter values'
    )

    parser.add_option(
        '--train-rank',
        dest = 'train_rank',
        default = '70',
        type = 'int',
        help = 'hyperparameter - rank'
    )

    parser.add_option(
        '--max-iter',
        dest = 'max_iter',
        default = '10',
        type = 'int',
        help = 'hyperparameter - max iter'
    )

    parser.add_option(
        '--reg-param',
        dest = 'reg_param',
        default = '0.1',
        type = 'float',
        help = 'hyperparameter - regularization parameter'
    )

    return parser