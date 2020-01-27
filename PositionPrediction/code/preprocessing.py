import os
import pickle
import glob

import pandas as pd
import numpy as np


class PreProcess:
    '''
    Class contains function to treat anomalies, outliers, missing and normalize sensor readings from train and test data.
    '''
    
    def __init__(self, load_cell_theshold, weight_threshold, outliers_threshold, \
                 map_sensors, target_map, position_to_remove, \
                 sensor_details_file, norm_sensor_details_file, model_folder):
        
        self.load_cell_theshold = load_cell_theshold  # setting the load cell threshold
        self.weight_threshold = weight_threshold  # setting the total weight threshold
        self.outliers_threshold = outliers_threshold  # setting the outliers standard deviation threshold
        self.map_sensors = map_sensors  # setting the map for old column names to new column names
        self.target_map = target_map  # mapping of position codes to 
        self.position_to_remove = position_to_remove  # samples with position to remove
        self.sensor_details_file = sensor_details_file  # file name for sensor details
        self.norm_sensor_details_file = norm_sensor_details_file  # file name for normalized sensor details
        self.model_folder = model_folder  # folder to save sensor details file
        
    # rename old names of planks to new names
    
    def rename_columns_if_needed(self, df):
        
        '''
        Function to rename old sensor names to new sensor names if needed.
        The old sensor names are like plank1_1a, plank1_1b, etc.
        The new sensor names are from LC1 to LC16
        '''
        
        # old column names are present in the data
        
        if len(set(df.columns).intersection(set(self.map_sensors.keys())))>0:
            
            # rename the old column names
            
            df.rename(columns=self.map_sensors, inplace=True)
        else:
            
            # do nothing if new column names are present
            
            pass
        
        return df
    
    # removing records where there are missing values in the subject column
    
    def rem_missing_train(self, df):
        
        '''
        Function to remove rows with missing values from train dataset and also to 
        save mean value of sensors for imputation on test dataset
        '''
        
    	# drop records with missing values in column
        
        sensors_list = [x for x in df.columns if 'LC' in x]
        
        df.dropna(how = 'any', subset = sensors_list, inplace = True)
        
    	# reset_index 
        
        df.reset_index(drop = True, inplace = True)
        
        # get the mean values of sensors and store in file for imputation in test data
        
        list_for_sensor_details = []
        
        for col in sensors_list:
            mean_val = df[col].mean()
            sensor_detail = {}
            sensor_detail['name'] = col
            sensor_detail['mean_val'] = mean_val
            list_for_sensor_details.append(sensor_detail)
            
        pickle.dump(list_for_sensor_details, open(os.path.join(self.model_folder, self.sensor_details_file), 'wb'))
        
        return df
    
    def treat_missing_test(self, df):
        
        '''
        Function to impute missing values in sensor readings for test data
        '''
        
    	# put mean values in column if the value is missing
        
        sensor_details_file = max(glob.glob(self.model_folder + '/*_sensor_details.pkl'), key=os.path.getctime)
        
        list_for_sensor_details = pickle.load(open(sensor_details_file, 'rb'))
        
        sensors_list = [x for x in df.columns if 'LC' in x]
        
        for i in range(len(sensors_list)):
            col = sensors_list[i]
            mean_val = list_for_sensor_details[i]['mean_val']
            df[col].fillna(mean_val, inplace = True)
            
        return df
    
    # removing records with load cell readings greater than load cell threshold
    
    def rem_load_cell_threshold(self, df):
        
        '''
        Function to remove samples with sensor readings greater than a particular threshold for train dataset
        '''
        
        #sensors list
        
        sensors_list = [x for x in df.columns if 'LC' in x]
    
    	# get indexes of samples where a particular sensor reading is greater than threshold
        
        index_for_sensors_greater_than_threshold = []
        
        for col in sensors_list:
    	    anomaly_indexes = df.loc[df[col]>self.load_cell_theshold].index
            
            # keep the indexes in the list
            
    	    index_for_sensors_greater_than_threshold.extend(anomaly_indexes)
    
    	# get the non outliers index
        
        non_outliers_index = [x for x in df.index if x not in index_for_sensors_greater_than_threshold]
    	
        # keep the non outliers index
        
        df = df.loc[non_outliers_index]
        
    	# reset_index 
        
        df.reset_index(drop = True, inplace = True)
        return df
    
    # remove records whose total weight is less than weight threshold
    
    def rem_less_weights(self, df):
        
        '''
        Function to remove samples with cumulative sensor readings 
        lower than a particular threshold from train dataset
        '''
        
        #sensors list
        
        sensors_list = [x for x in df.columns if 'LC' in x]
        
        # create a column containing of the sensors readings
        
        df['sum_sensors'] = df[sensors_list].sum(axis = 1)
        
        # remove those indices
        
        df = df.loc[df['sum_sensors']>self.weight_threshold]
        
        # drop the sum_sensors column
        
        df.drop(['sum_sensors'], axis = 1, inplace = True)
        
        # reset_index 
        
        df.reset_index(drop = True, inplace = True)
        
        return df
    
    # remove records with sitting positions
    
    def rem_sitting(self, df):
        
        '''
        Function to remove samples with sitting position
        '''
        
        # get the sitting position index 
        
        sitting_position_index = list(self.target_map.keys())[list(self.target_map.values()).index(self.position_to_remove)]
        
        # remove those indices
        
        df = df.loc[df['manual_position']!=sitting_position_index]
    	
        # reset_index 
        
        df.reset_index(drop = True, inplace = True)
        return df
    
    # normalize the data
    
    def normalize(self, df):
        
        '''
        Function to normalize the sensor readings for both train and test data
        '''        
        
        # get all the columns of dataframe
        
        cols = df.columns
        
        # get the columns containing sensor readings
        
        sensors_list = [x for x in df.columns if 'LC' in x]
        sensors_pct_list = [x + '_pct' for x in sensors_list]
        
        # create an empty dataframe
        
        percent_df = pd.DataFrame()
        
        # fill the dataframe
        
        for i, s in enumerate(sensors_pct_list):
            percent_df[s] = np.round(df[sensors_list[i]]/df[sensors_list].sum(axis = 1), 4) * 100
            
        # get all the other columns
        
        other_cols = [col for col in cols if col not in sensors_list]
        
        # insert the data of other columns into normalized dataframe
        
        for col in other_cols:
            percent_df[col] = df[col]
        
        return percent_df
    
    # remove outliers based on 3.5 standard deviation away from mean from train
    
    def treat_outliers_train(self, percent_df):
        
        '''
        Function to remove samples with outliers from train dataset
        '''
        
        # no. of standard deviation away from mean
        
        std_val = self.outliers_threshold
        
        # sensor names with pct
        
        sensors_pct_list = [x for x in percent_df.columns if 'LC' in x]
            
        # list to contain the outliers index
        
        index_list_for_outliers = []
            
        # list to contain sensor details 
            
        list_for_sensor_details = []
        
        for col in sensors_pct_list:
                
            # individual sensor details values
                
            sensor_detail = {}
                
            # mean value for a particular sensor
            
            mean_sensor = percent_df[col].mean()
            
            # standard deviation for a particular 
            
            std_sensor = percent_df[col].std()
            
            # upper limit
            
            upper_threshold = mean_sensor+(std_sensor * std_val)
            
            # lower limit
            
            lower_threshold = mean_sensor-(std_sensor * std_val)
            
            # outliers indexes for a particular sensor
            
            outliers_index = percent_df.loc[(percent_df[col]<lower_threshold)|(percent_df[col]>upper_threshold)].index 
                                            
            # outliers index with duplicates
            
            index_list_for_outliers.extend(list(outliers_index))
                
            sensor_detail['name'] = col
            sensor_detail['mean_val'] = mean_sensor
            sensor_detail['std_val'] = std_sensor
            list_for_sensor_details.append(sensor_detail)
                
        # save the sensor details file
            
        pickle.dump(list_for_sensor_details, open(os.path.join(self.model_folder, self.norm_sensor_details_file), 'wb'))
                
        # index list for outliers for all the sensors without duplicates
            
        index_list_for_outliers = list(set(index_list_for_outliers))
            
        # non outliers indexes
            
        non_outliers_index = [x for x in percent_df.index if x not in index_list_for_outliers]
            
        # remove outliers index from the dataframe
            
        percent_df = percent_df.loc[non_outliers_index]
        
        # reset the index of the dataframe
        
        percent_df.reset_index(drop=True, inplace=True)
        return percent_df
    
    # treat outliers based on 3.5 standard deviation away from mean from test
    
    def treat_outliers_test(self, percent_df):
        
        '''
        Function to treat outlier values from test dataset.
        The outliers will be capped at upper and lower threshold which are 
        3.5 standard deviation away from mean.
        '''
        
        # no. of standard deviation away from mean
        
        std_val = self.outliers_threshold
        
        # sensor names with pct
        
        sensors_pct_list = [x for x in percent_df.columns if 'LC' in x]
            
        # list that contains sensor details 
        
        norm_sensor_details_file = max(glob.glob(self.model_folder + '/*_norm_sensor_details.pkl'), key=os.path.getctime)
        
        list_for_sensor_details = pickle.load(open(norm_sensor_details_file, 'rb'))
        
        for i in range(len(sensors_pct_list)):
            
            col = sensors_pct_list[i]
            
            # mean value of the sensor
            
            mean_sensor = list_for_sensor_details[i]['mean_val']
            
            # standard deviation of the sensor 
            
            std_sensor = list_for_sensor_details[i]['std_val']
            
            # lower_threshold 
            
            lower_threshold = mean_sensor - (std_sensor * std_val)
            
            # upper threshold
            
            upper_threshold = mean_sensor + (std_sensor * std_val)
            
            # cap all the values beyond lower threshold to lower threshold
            
            percent_df.loc[percent_df[col]<lower_threshold, col] = lower_threshold
            
            # cap all the values beyond upper threshold to upper threshold
            
            percent_df.loc[percent_df[col]>upper_threshold, col] = upper_threshold
            
            #percent_df = percent_df.loc[(percent_df[col]>=lower_threshold) & (percent_df[col]<=upper_threshold)]
            #percent_df.reset_index(drop=True, inplace=True)
        return percent_df