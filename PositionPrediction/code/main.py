import sys
import logging
import time
import socket
from pandas import read_csv, DataFrame

from preprocessing import PreProcess
from feature_engineering import FeatureExtractor
from learner import Learner
from classifier import Classifier
import config

start = time.time()

# import numpy as np
# from sklearn.metrics import accuracy_score, log_loss

# importing log information

log_folder = config.LOGS_FOLDER

date_time_format = config.DATETIME_FORMAT
log_level = config.LOG_LEVEL

# logger details

# logging.basicConfig(level = log_level, \
# filename = log_folder + '/' + time.strftime(date_time_format) + '_learning.log', \
# filemode='w' ,format='%(asctime)s %(levelname)s : %(message)s', \
# datefmt='%m/%d/%Y %I:%M:%S %p')

# model name

model_name = time.strftime(date_time_format) + '_model.sav'

# file for sensor details

sensor_details_file = time.strftime(date_time_format) + '_sensor_details.pkl'

# file for normalized sensor details

norm_sensor_details_file = time.strftime(date_time_format) + '_norm_sensor_details.pkl'


class FitAndPredict:
    """
    Class contains function for the training and classification pipeline
    """

    def __init__(self):
        self.train_file = config.TRAIN_FILE
        self.test_file = config.TEST_FILE
        self.predicted_test_file = config.PREDICTED_TEST_FILE
        self.model_folder = config.MODELS_FOLDER
        self.target_map = config.TARGET_MAP
        self.map_sensors = config.MAP_SENSORS
        self.load_cell_theshold = config.LOAD_CELL_THRESHOLD
        self.weight_threshold = config.WEIGHT_THRESHOLD
        self.outliers_threshold = config.OUTLIERS_THRESHOLD
        self.feature_names = config.FEATURE_NAMES
        self.dev_map = config.DEV_MAP
        self.plank_dict = config.PLANK_DICT
        self.position_to_remove = config.POSITION_TO_REMOVE
        self.sensor_details_file = sensor_details_file
        self.norm_sensor_details_file = norm_sensor_details_file
        self.random_state = config.RANDOM_STATE
        self.min_samples_split = config.MIN_SAMPLES_SPLIT
        self.min_samples_leaf = config.MIN_SAMPLES_LEAF
        self.n_estimators = config.N_ESTIMATORS
        self.model_name = model_name
        self.target_column = config.TARGET_COLUMN
        self.preprocess = PreProcess(self.load_cell_theshold, self.weight_threshold, self.outliers_threshold,
                                     self.map_sensors, self.target_map, self.position_to_remove,
                                     self.sensor_details_file, self.norm_sensor_details_file, self.model_folder)
        self.fe = FeatureExtractor(self.plank_dict)

    def read_train_data(self):

        """
        Function to read the train data in the training pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start read_train_data()')
        try:
            self.input_data = read_csv(self.train_file)
            logging.debug(__name__ + ' shape : ' + str(self.input_data.shape))
            logging.debug(__name__ + ' : ' + ' End read_train_data()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Input file not found ')
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

    def check_train_data(self):

        """
        Function to check if all load cell columns and target column are present in the train data
        """

        logging.debug(__name__ + ' : ' + ' Start check_train_data()')
        try:
            train_columns = self.input_data.columns.values
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # check if all load cell columns are present in the train data

        try:
            if (set(self.map_sensors.keys()).issubset(set(train_columns))) or \
                    (set(self.map_sensors.values()).issubset(set(train_columns))):
                pass
            else:
                print("LOAD CELL COLUMNS NOT PRESENT IN TRAIN DATA")
                logging.debug(__name__ + ' : ' + ' LOAD CELL COLUMNS NOT PRESENT IN TRAIN DATA')
                logging.debug(__name__ + ' : ' + ' End read_train_data()')
                sys.exit()
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # check if target column is present in the data

        try:
            if (set([self.target_column]).issubset(set(train_columns))):
                pass
            else:
                print("TARGET COLUMN NOT PRESENT IN TRAIN DATA")
                logging.debug(__name__ + ' : ' + ' TARGET COLUMN NOT PRESENT IN TRAIN DATA')
                logging.debug(__name__ + ' : ' + ' End read_train_data()')
                sys.exit()
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # check if all 5 positions are present in target column of train data

        try:
            if (set(self.input_data[self.target_column].unique()).issubset(set([1, 2, 3, 4, 5]))):
                pass
            else:
                print("VALUES OTHER THAN PRESPECIFIED POSITION VALUES PRESENT IN TRAIN DATA")
                logging.debug(
                    __name__ + ' : ' + ' VALUES OTHER THAN PRESPECIFIED POSITION VALUES PRESENT IN TRAIN DATA')
                logging.debug(__name__ + ' : ' + ' End read_train_data()')
                sys.exit()
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        logging.debug(__name__ + ' : ' + ' End check_train_data()')

        return

    def preprocess_train_data(self):

        """
        Function to preprocess the train data in the training pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start preprocess_train_data()')

        try:
            logging.debug(__name__ + ' : ' + ' Start rename_columns_if_needed()')
            self.preprocessed_input_data = self.preprocess.rename_columns_if_needed(self.input_data)
            logging.debug(__name__ + ' : ' + ' End rename_columns_if_needed()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start rem_missing_train()')
            self.preprocessed_input_data = self.preprocess.rem_missing_train(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End rem_missing_train()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start rem_load_cell_threshold()')
            self.preprocessed_input_data = self.preprocess.rem_load_cell_threshold(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End rem_load_cell_threshold()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start rem_sitting()')
            self.preprocessed_input_data = self.preprocess.rem_sitting(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End rem_sitting()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start normalize()')
            self.preprocessed_input_data = self.preprocess.normalize(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End normalize()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start treat_outliers_train()')
            self.preprocessed_input_data = self.preprocess.treat_outliers_train(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End treat_outliers_train()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start rem_less_weights()')
            self.preprocessed_input_data = self.preprocess.rem_less_weights(self.preprocessed_input_data)
            logging.debug(__name__ + ' : ' + ' End rem_less_weights()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        logging.debug(__name__ + ' shape : ' + str(self.preprocessed_input_data.shape))
        logging.debug(__name__ + ' : ' + ' End preprocess_train_data()')

        return

    def read_test_data(self):

        """
        Function to read the test data in the classification pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start read_test_data()')
        try:
            self.test_data = read_csv(self.test_file, header=None)
            logging.debug(__name__ + ' shape : ' + str(self.test_data.shape))
            logging.debug(__name__ + ' : ' + ' End read_test_data()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Test file not found ')
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            return

    def check_test_data(self):

        """
        Function to check if all load cell columns are present in the test data
        """
        logging.debug(__name__ + ' : ' + ' Start check_test_data()')

        # check if all load cell columns are present in the test data
        # try:
        # test_columns = self.test_data.columns
        # except Exception as e:
        # logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        # finally:
        # pass
        try:
            # if (set(self.map_sensors.keys()).issubset(set(test_columns))) or (set(self.map_sensors.values()).issubset(set(test_columns))):
            test_columns = ['LC' + str(x) for x in range(1, 17)]
            if self.test_data.shape[1] == len(test_columns):
                self.test_data.columns = test_columns
                pass
            else:
                print("TEST DATA DO NOT HAVE ALL LOAD CELLS DATA")
                logging.debug(__name__ + ' : ' + ' TEST DATA DO NOT HAVE ALL LOAD CELLS DATA')
                logging.debug(__name__ + ' : ' + ' End check_test_data()')
                sys.exit()
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        logging.debug(__name__ + ' : ' + ' End check_test_data()')

        return

    def preprocess_test_data(self):

        """
        Function to preprocess the test data in the classification pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start preprocess_test_data()')

        try:
            self.preprocessed_test_data = self.test_data
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # try:
        #    logging.debug(__name__ + ' : ' + ' Start rename_columns_if_needed()')
        #    self.preprocessed_test_data = self.preprocess.rename_columns_if_needed(self.test_data)
        #    logging.debug(__name__ + ' : ' + ' End rename_columns_if_needed()')
        # except Exception as e:
        #    logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        # finally:
        #    pass

        try:
            logging.debug(__name__ + ' : ' + ' Start treat_missing_test()')
            self.preprocessed_test_data = self.preprocess.treat_missing_test(self.preprocessed_test_data)
            logging.debug(__name__ + ' : ' + ' End treat_missing_test()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start normalize()')
            self.preprocessed_test_data = self.preprocess.normalize(self.preprocessed_test_data)
            logging.debug(__name__ + ' : ' + ' End normalize()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            logging.debug(__name__ + ' : ' + ' Start treat_outliers_test()')
            self.preprocessed_test_data = self.preprocess.treat_outliers_test(self.preprocessed_test_data)
            logging.debug(__name__ + ' : ' + ' End treat_outliers_test()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # logging.debug(__name__ + ' shape : ' + str(self.preprocessed_test_data.shape))
        # logging.debug(__name__ + ' : ' + ' End preprocess_test_data()')

        return

    def transform_train_data_into_features(self):

        """
        Function to create features from train data in the training pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start transform_train_data_into_features()')

        try:
            # left_sensors_pct
            logging.debug(__name__ + ' : ' + ' Start left_percent()')
            self.preprocessed_input_data['left_sensors_pct'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.left_percent(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End left_percent()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            #  plank_1_std
            logging.debug(__name__ + ' : ' + ' Start plank_1_std_cal()')
            self.preprocessed_input_data['plank_1_std'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.plank_1_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_1_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_2_std
            logging.debug(__name__ + ' : ' + ' Start plank_2_std_cal()')
            self.preprocessed_input_data['plank_2_std'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.plank_2_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_2_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_std
            logging.debug(__name__ + ' : ' + ' Start plank_3_std_cal()')
            self.preprocessed_input_data['plank_3_std'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.plank_3_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_3_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_std
            logging.debug(__name__ + ' : ' + ' Start plank_4_std_cal()')
            self.preprocessed_input_data['plank_4_std'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.plank_4_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_4_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_1_x
            logging.debug(__name__ + ' : ' + ' Start get_com_1_x()')
            self.preprocessed_input_data['plank_1_com_x'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_1_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_1_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_2_x
            logging.debug(__name__ + ' : ' + ' Start get_com_2_x()')
            self.preprocessed_input_data['plank_2_com_x'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_2_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_2_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_3_x
            logging.debug(__name__ + ' : ' + ' Start get_com_3_x()')
            self.preprocessed_input_data['plank_3_com_x'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_3_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_3_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_4_x
            logging.debug(__name__ + ' : ' + ' Start get_com_4_x()')
            self.preprocessed_input_data['plank_4_com_x'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_4_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_4_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_1_y
            logging.debug(__name__ + ' : ' + ' Start get_com_1_y()')
            self.preprocessed_input_data['plank_1_com_y'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_1_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_1_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_2_y
            logging.debug(__name__ + ' : ' + ' Start get_com_2_y()')
            self.preprocessed_input_data['plank_2_com_y'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_2_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_2_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_3_y
            logging.debug(__name__ + ' : ' + ' Start get_com_3_y()')
            self.preprocessed_input_data['plank_3_com_y'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_3_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_3_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_4_y
            logging.debug(__name__ + ' : ' + ' Start get_com_4_y()')
            self.preprocessed_input_data['plank_4_com_y'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_com_4_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_4_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # y_errors
            logging.debug(__name__ + ' : ' + ' Start get_errors_from_fitted_line()')
            self.preprocessed_input_data['y_errors'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_errors_from_fitted_line(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_errors_from_fitted_line()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_deviation from fitted line through COMs of first two planks
            logging.debug(__name__ + ' : ' + ' Start get_deviation_plank_3()')
            self.preprocessed_input_data['plank_3_dev'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_deviation_plank_3(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_deviation_plank_3()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_deviation from fitted line through COMs of first two planks
            logging.debug(__name__ + ' : ' + ' Start get_deviation_plank_4()')
            self.preprocessed_input_data['plank_4_dev'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.get_deviation_plank_4(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_deviation_plank_4()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_dev_bucket
            logging.debug(__name__ + ' : ' + ' Start bucketize_plank_3_dev()')
            self.preprocessed_input_data['plank_3_dev_bucket'] = self.preprocessed_input_data['plank_3_dev'].apply(
                lambda x: self.fe.bucketize_plank_dev(x))
            self.preprocessed_input_data['plank_3_dev_bucket'] = self.preprocessed_input_data[
                'plank_3_dev_bucket'].apply(lambda x: self.dev_map[x])
            logging.debug(__name__ + ' : ' + ' End bucketize_plank_3_dev()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_dev_bucket
            logging.debug(__name__ + ' : ' + ' Start bucketize_plank_4_dev()')
            self.preprocessed_input_data['plank_4_dev_bucket'] = self.preprocessed_input_data['plank_4_dev'].apply(
                lambda x: self.fe.bucketize_plank_dev(x))
            self.preprocessed_input_data['plank_4_dev_bucket'] = self.preprocessed_input_data[
                'plank_4_dev_bucket'].apply(lambda x: self.dev_map[x])
            logging.debug(__name__ + ' : ' + ' End bucketize_plank_4_dev()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_deviation from fitted line through COMs of 2nd and 3rd planks
            logging.debug(__name__ + ' : ' + ' Start plank_4_wrt_3_2()')
            self.preprocessed_input_data['plank_4_wrt_3_2'] = self.preprocessed_input_data.apply(
                lambda x: self.fe.plank_4_wrt_3_2(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_4_wrt_3_2()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        logging.debug(__name__ + ' shape : ' + str(self.preprocessed_input_data.shape))
        logging.debug(__name__ + ' : ' + ' End transform_train_data_into_features()')

        return

    def transform_test_data_into_features(self):

        """
        Function to create features from test data in the classification pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start transform_test_data_into_features()')

        try:
            # left_sensors_pct
            logging.debug(__name__ + ' : ' + ' Start left_percent()')
            self.preprocessed_test_data['left_sensors_pct'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.left_percent(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End left_percent()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_1_std
            logging.debug(__name__ + ' : ' + ' Start plank_1_std_cal()')
            self.preprocessed_test_data['plank_1_std'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.plank_1_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_1_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_2_std
            logging.debug(__name__ + ' : ' + ' Start plank_2_std_cal()')
            self.preprocessed_test_data['plank_2_std'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.plank_2_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_2_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_std
            logging.debug(__name__ + ' : ' + ' Start plank_3_std_cal()')
            self.preprocessed_test_data['plank_3_std'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.plank_3_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_3_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_std
            logging.debug(__name__ + ' : ' + ' Start plank_4_std_cal()')
            self.preprocessed_test_data['plank_4_std'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.plank_4_std_cal(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_4_std_cal()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_1_x
            logging.debug(__name__ + ' : ' + ' Start get_com_1_x()')
            self.preprocessed_test_data['plank_1_com_x'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_1_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_1_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_2_x
            logging.debug(__name__ + ' : ' + ' Start get_com_2_x()')
            self.preprocessed_test_data['plank_2_com_x'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_2_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_2_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_3_x
            logging.debug(__name__ + ' : ' + ' Start get_com_3_x()')
            self.preprocessed_test_data['plank_3_com_x'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_3_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_3_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_4_x
            logging.debug(__name__ + ' : ' + ' Start get_com_4_x()')
            self.preprocessed_test_data['plank_4_com_x'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_4_x(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_4_x()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_1_y
            logging.debug(__name__ + ' : ' + ' Start get_com_1_y()')
            self.preprocessed_test_data['plank_1_com_y'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_1_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_1_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_2_y
            logging.debug(__name__ + ' : ' + ' Start get_com_2_y()')
            self.preprocessed_test_data['plank_2_com_y'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_2_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_2_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_3_y
            logging.debug(__name__ + ' : ' + ' Start get_com_3_y()')
            self.preprocessed_test_data['plank_3_com_y'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_3_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_3_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # com_4_y
            logging.debug(__name__ + ' : ' + ' Start get_com_4_y()')
            self.preprocessed_test_data['plank_4_com_y'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_com_4_y(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_com_4_y()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # y_errors
            logging.debug(__name__ + ' : ' + ' Start get_errors_from_fitted_line()')
            self.preprocessed_test_data['y_errors'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_errors_from_fitted_line(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_errors_from_fitted_line()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_deviation from fitted line through COMs of first two planks
            logging.debug(__name__ + ' : ' + ' Start get_deviation_plank_3()')
            self.preprocessed_test_data['plank_3_dev'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_deviation_plank_3(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_deviation_plank_3()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_deviation from fitted line through COMs of first two planks
            logging.debug(__name__ + ' : ' + ' Start get_deviation_plank_4()')
            self.preprocessed_test_data['plank_4_dev'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.get_deviation_plank_4(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End get_deviation_plank_4()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_3_dev_bucket
            logging.debug(__name__ + ' : ' + ' Start bucketize_plank_3_dev()')
            self.preprocessed_test_data['plank_3_dev_bucket'] = self.preprocessed_test_data['plank_3_dev'].apply(
                lambda x: self.fe.bucketize_plank_dev(x))
            self.preprocessed_test_data['plank_3_dev_bucket'] = self.preprocessed_test_data['plank_3_dev_bucket'].apply(
                lambda x: self.dev_map[x])
            logging.debug(__name__ + ' : ' + ' End bucketize_plank_3_dev()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_dev_bucket
            logging.debug(__name__ + ' : ' + ' Start bucketize_plank_4_dev()')
            self.preprocessed_test_data['plank_4_dev_bucket'] = self.preprocessed_test_data['plank_4_dev'].apply(
                lambda x: self.fe.bucketize_plank_dev(x))
            self.preprocessed_test_data['plank_4_dev_bucket'] = self.preprocessed_test_data['plank_4_dev_bucket'].apply(
                lambda x: self.dev_map[x])
            logging.debug(__name__ + ' : ' + ' End bucketize_plank_4_dev()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        try:
            # plank_4_deviation from fitted line through COMs of 2nd and 3rd planks
            logging.debug(__name__ + ' : ' + ' Start plank_4_wrt_3_2()')
            self.preprocessed_test_data['plank_4_wrt_3_2'] = self.preprocessed_test_data.apply(
                lambda x: self.fe.plank_4_wrt_3_2(x), axis=1)
            logging.debug(__name__ + ' : ' + ' End plank_4_wrt_3_2()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            pass

        # logging.debug(__name__ + ' shape : ' + str(self.preprocessed_test_data.shape))
        # logging.debug(__name__ + ' : ' + ' End transform_test_data_into_features()')

        return

    def train_model(self):

        """
        Function to train the model on train data for training pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start train_model()')

        try:
            # train the model
            self.learner = Learner(self.n_estimators, self.min_samples_split, self.min_samples_leaf, self.random_state,
                                   self.model_folder, self.model_name)
            X_train = self.preprocessed_input_data[self.feature_names]
            Y_train = self.preprocessed_input_data[self.target_column]
            self.learner.train_model(X_train, Y_train)
            # predictions, pred_prob
            logging.debug(__name__ + ' : ' + ' End train_model()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            return

    def classify(self):

        """
        Function to make classifications on test data for the classification pipeline
        """

        logging.debug(__name__ + ' : ' + ' Start classify()')

        try:

            # classify using the model
            self.classifier = Classifier(self.model_folder)
            X_test = self.preprocessed_test_data[self.feature_names]
            # Y_test = self.preprocessed_test_data[self.target_column]
            # print (X_test.shape, Y_test.shape)
            predictions, pred_prob = self.classifier.classify_model(X_test)

            # saving the test dataset with the predicted values

            # self.test_data[self.target_column] = predictions
            # self.test_data.to_csv(self.predicted_test_file, index = False)
            self.predicted_test_data = DataFrame({self.target_column: predictions})
            self.predicted_test_data.to_csv(self.predicted_test_file, index=False, header=False)

            # printing the accuracy score
            # print (np.round(accuracy_score(Y_test, predictions), 4) * 100)
            # print (np.round(log_loss(Y_test, pred_prob), 2))

            logging.debug(__name__ + ' : ' + ' End classify()')
        except Exception as e:
            logging.error(__name__ + ' : ' + ' Error: ' + str(e))
        finally:
            return


'''        
if __name__ == '__main__':
    fit_and_predict = FitAndPredict()
    fit_and_predict.read_train_data()
    #fit_and_predict.preprocess_train_data()
    fit_and_predict.transform_train_data_into_features()
    fit_and_predict.train_model()
    fit_and_predict.read_test_data()
    #fit_and_predict.preprocess_test_data()
    fit_and_predict.transform_test_data_into_features()
    fit_and_predict.classify() 
'''


def main():
    fit_and_predict = FitAndPredict()
    s = socket.socket()
    s.bind(('', 32017))

    #    sndmq = sysv_ipc.MessageQueue(54321, sysv_ipc.IPC_CREAT)
    arg_train_test = sys.argv[1]
    if arg_train_test.strip() == "train":
        print("TRAINING STARTED ...")
        print("Reading Train Data ...")
        fit_and_predict.read_train_data()
        print("Checking Train Data ...")
        fit_and_predict.check_train_data()
        print("Preprocessing Train Data ...")
        fit_and_predict.preprocess_train_data()
        print("Creating Features from Train Data ...")
        fit_and_predict.transform_train_data_into_features()
        print("Training model on Train Data ...")
        fit_and_predict.train_model()
        print("TRAINING ENDED ...")
        end = time.time()
        time_elapsed = end - start
        logging.info(__name__ + ' : ' + ' TIme Elapsed: ' + str(time_elapsed))
    elif arg_train_test.strip() == "test":
        while 1:
            s.listen(5)
            print("Entering While Loop...")
            c, addr = s.accept()
            print("Got client connection")

            # print(value)
            #            byte = 'a'
            #            mq.send(byte, False, 1)
            print("CLASSIFICATION STARTED ...")
            print("Reading Test Data ...")
            fit_and_predict.read_test_data()
            print("Checking Test Data ...")
            fit_and_predict.check_test_data()
            print("Preprocessing Test Data ...")
            fit_and_predict.preprocess_test_data()
            print("Creating Features from Test Data ...")
            fit_and_predict.transform_test_data_into_features()
            print("Classifying Test Data ...")
            fit_and_predict.classify()
            print("CLASSIFICATION ENDED ...")
            end = time.time()
            time_elapsed = end - start
            logging.info(__name__ + ' : ' + ' TIme Elapsed: ' + str(time_elapsed))
            c.send(b'Received')
            c.close()
    else:
        print("ENTER A VALID ARGUMENT")


if __name__ == '__main__':
    main()
