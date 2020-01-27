import logging
import os

# current directory

cur_dir = os.getcwd()

# parent directory

PARENT_FOLDER = os.path.abspath(os.path.join(cur_dir, './'))
#print ("Parent", PARENT_FOLDER)

# train folder

TRAIN_FOLDER = os.path.abspath(os.path.join(PARENT_FOLDER, 'data_train'))

# train preprocessed folder

#TRAIN_PREPROCESSED_FOLDER = os.path.abspath(os.path.join(TRAIN_FOLDER, 'preprocessed'))

# train

TRAIN_FILE_NAME = 'train_data_20190314.csv'
TRAIN_FILE = os.path.abspath(os.path.join(TRAIN_FOLDER, TRAIN_FILE_NAME))

# test folder

TEST_FOLDER = os.path.abspath(os.path.join(PARENT_FOLDER, 'data_classify'))

# test preprocessed folder

#TEST_PREPROCESSED_FOLDER = os.path.abspath(os.path.join(TEST_FOLDER, 'preprocessed'))

# test

TEST_FILE_NAME = 'test_data.csv'
TEST_FILE = os.path.abspath(os.path.join(TEST_FOLDER, TEST_FILE_NAME))

PREDICTED_TEST_FILE_NAME = 'predicted_' + TEST_FILE_NAME
PREDICTED_TEST_FILE = os.path.abspath(os.path.join(TEST_FOLDER, PREDICTED_TEST_FILE_NAME))

# logs folder

LOGS_FOLDER = os.path.abspath(os.path.join(PARENT_FOLDER, 'log'))
#print (LOGS_FOLDER)

# models folder

MODELS_FOLDER = os.path.abspath(os.path.join(PARENT_FOLDER, 'model'))

# information of each plank

PLANK_DICT = [
    {
        'name':'Plank 1', 
        'left_sensors':['LC2', 'LC3'],
        'right_sensors':['LC1', 'LC4'],
        'size':(460, 460)
    },
    {
        'name':'Plank 2', 
        'left_sensors':['LC6', 'LC7'],
        'right_sensors':['LC5', 'LC8'],
        'size':(69, 460)
    },
    {
        'name':'Plank 3', 
        'left_sensors':['LC10', 'LC11'],
        'right_sensors':['LC9', 'LC12'],
        'size':(125, 460)
    },
    {
        'name':'Plank 4', 
        'left_sensors':['LC14', 'LC15'],
        'right_sensors':['LC13', 'LC16'],
        'size':(250, 460)
    }
]

# map of encoded positions to actual positions

TARGET_MAP = {
    1 : 'Left Lateral',
    2 : 'Supine',
    3 : 'Right Lateral',
    4 : 'Partial Left',
    5 : 'Partial Right',
    6 : 'Sitting'
    }

# map of old sensor names to new sensor names

MAP_SENSORS = {
	'plank1_1a' : 'LC1', 
	'plank1_1b' : 'LC2', 
    'plank1_2a' : 'LC3', 
	'plank1_2b' : 'LC4', 
    'plank2_1a' : 'LC5', 
    'plank2_1b' : 'LC6', 
    'plank2_2a' : 'LC7', 
    'plank2_2b' : 'LC8', 
    'plank3_1a' : 'LC9', 
    'plank3_1b' : 'LC10', 
    'plank3_2a' : 'LC11', 
    'plank3_2b' : 'LC12', 
    'plank4_1a' : 'LC13', 
    'plank4_1b' : 'LC14', 
    'plank4_2a' : 'LC15', 
    'plank4_2b' : 'LC16'
    }

# feature names

FEATURE_NAMES = ['left_sensors_pct', 'plank_1_std', 'plank_2_std', 'plank_3_std', \
                 'plank_4_std', 'y_errors', 'plank_3_dev_bucket', 'plank_4_dev_bucket', \
                 'plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y']

# target column

TARGET_COLUMN = 'manual_position'

# preprocessing threshold

LOAD_CELL_THRESHOLD = 50  # to remove readings with load cell readings greater than load_cell_threshold
WEIGHT_THRESHOLD = 40  # remove readings whose total weight is less than weight threshold
OUTLIERS_THRESHOLD = 3.5  # remove outliers based on standard deviation threshold

# Encode plank_3_dev_bucket and plank_4_dev_bucket

DEV_MAP = {
    'positive_high':3,
    'on':2,
    'negative_high':1
    }

# logger

LOG_LEVEL=logging.DEBUG  # logging level
DATETIME_FORMAT = '%m%d%Y%H%M%S'  # date time format for log file creation

# random forest hyperparameters

N_ESTIMATORS = 20
MIN_SAMPLES_SPLIT = 25
MIN_SAMPLES_LEAF = 5
RANDOM_STATE = 12

# samples with position to remove

POSITION_TO_REMOVE = 'Sitting'