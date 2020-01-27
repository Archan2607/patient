from __future__ import division
import numpy as np
np.seterr(all='raise')
from sklearn.linear_model import LinearRegression
#import math	


lr = LinearRegression() # linear regression function to predict the COMs of planks


class FeatureExtractor:
    
    '''
    Class contains functions to create features from train and test data
    '''
    
    def __init__(self, plank_dict):
        self.plank_dict = plank_dict

    def left_percent(self, x):
        
        '''
        Function calculates the cumulative percentage of left sensors.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        left_sensors = ['LC2', 'LC3', 'LC6', 'LC7', 'LC10', 'LC11', 'LC14', 'LC15'] # list for left sensors
        left_sensors_pct_list = [x + '_pct' for x in left_sensors] # create a list for left sensors percent
    
        try:
            #sum the left sensor values
            left_sensors_pct = np.sum(x[left_sensors_pct_list].values)  
        except:
            left_sensors_pct = 0
        finally:
            return left_sensors_pct

    def plank_1_std_cal(self, x):
        
        '''
        Function calculates the standard deviation of plank 1.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        plank_1 =  ['LC1', 'LC2', 'LC3', 'LC4'] # for plank 1
        plank_1_pct = [x + '_pct' for x in plank_1]
        try:
            plank_1_std_val = np.std(x[plank_1_pct].values, ddof = 1)
        except:
            plank_1_std_val = 0
        finally:
            return plank_1_std_val

    def plank_2_std_cal(self, x):
        
        '''
        Function calculates the standard deviation of plank 2.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        plank_2 =  ['LC5', 'LC6', 'LC7', 'LC8'] # for plank 2
        plank_2_pct = [x + '_pct' for x in plank_2]
        try:
            plank_2_std_val = np.std(x[plank_2_pct].values, ddof = 1)
        except:
            plank_2_std_val = 0
        finally:
            return plank_2_std_val

    def plank_3_std_cal(self, x):
        
        '''
        Function calculates the standard deviation of plank 3.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        plank_3 =  ['LC9', 'LC10', 'LC11', 'LC12'] # for plank 3
        plank_3_pct = [x + '_pct' for x in plank_3]
        try:
            plank_3_std_val = np.std(x[plank_3_pct].values, ddof = 1)
        except:
            plank_3_std_val = 0
        finally:
            return plank_3_std_val

    def plank_4_std_cal(self, x):
        
        '''
        Function calculates the standard deviation of plank 4.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        plank_4 =  ['LC13', 'LC14', 'LC15', 'LC16'] # for plank 4
        plank_4_pct = [x + '_pct' for x in plank_4]
        try:
            plank_4_std_val = np.std(x[plank_4_pct].values, ddof = 1)
        except:
            plank_4_std_val = 0
        finally:
            return plank_4_std_val

    def get_com_1_x(self, x):
        
        '''
        Function calculates x co-ordinate of COM of plank 1.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(1, 5)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_x = self.calculate_com_x(plank_vals, plank_index = 0)
        except:
            com_x = self.plank_dict[0]['size'][0]/2
        finally:
            return com_x

    def get_com_2_x(self, x):
        
        '''
        Function calculates x co-ordinate of COM of plank 2.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(5, 9)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_x = self.calculate_com_x(plank_vals, plank_index = 1) + \
            self.plank_dict[0]['size'][0]
        except:
            com_x = self.plank_dict[0]['size'][0] + \
            self.plank_dict[1]['size'][0]/2
        finally:
            return com_x

    def get_com_3_x(self, x):
        
        '''
        Function calculates x co-ordinate of COM of plank 3.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(9, 13)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_x = self.calculate_com_x(plank_vals, plank_index = 2) + \
            self.plank_dict[0]['size'][0] + \
            self.plank_dict[1]['size'][0]
        except:
            com_x = self.plank_dict[0]['size'][0] + \
            self.plank_dict[1]['size'][0] + \
            self.plank_dict[2]['size'][0]/2
        finally:
            return com_x

    def get_com_4_x(self, x):
        
        '''
        Function calculates x co-ordinate of COM of plank 4.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(13, 17)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_x = self.calculate_com_x(plank_vals, plank_index = 3) + \
            self.plank_dict[0]['size'][0] + \
            self.plank_dict[1]['size'][0] + \
            self.plank_dict[2]['size'][0]
        except:
            com_x = self.plank_dict[0]['size'][0] + \
            self.plank_dict[1]['size'][0] + \
            self.plank_dict[2]['size'][0] + \
            self.plank_dict[3]['size'][0]/2
        finally:
            return com_x

    def calculate_com_x(self, x, plank_index):
        
        '''
        Function takes the values of a particular and 
        calculates the x co-ordinate of COM for that particular plank.
        '''
        
        # first left sensor value of plank
        left_val_1 = x[1]
        # second left sensor value of plank
        left_val_2 = x[2]
        # first right sensor value of plank
        right_val_1 = x[0]
        # second right sensor value of plank
        right_val_2 = x[3]
        # total weight percentage of the plank
        total_weight = np.sum(x)
        # for calculating the x cordinate of centre of mass of plank
        com_plank_x = ((right_val_1 *  0) + (left_val_1 * 0) + \
            (right_val_2 * self.plank_dict[plank_index]['size'][0]) + \
            (left_val_2 * self.plank_dict[plank_index]['size'][0]))/total_weight
    
        return np.int(com_plank_x)

    def get_com_1_y(self, x):
        
        '''
        Function calculates y co-ordinate of COM of plank 1.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(1, 5)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_y = self.calculate_com_y(plank_vals, plank_index = 0)
        except:
            com_y = 0.5 # mid point of the bed
        finally:
            return com_y
    
    def get_com_2_y(self, x):
        
        '''
        Function calculates y co-ordinate of COM of plank 2.
        Input (x) refers to a sample with its corresponding values
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(5, 9)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_y = self.calculate_com_y(plank_vals, plank_index = 1)
        except:
            com_y = 0.5 # mid point of the bed
        finally:
            return com_y
    
    def get_com_3_y(self, x):
        
        '''
        Function calculates y co-ordinate of COM of plank 3.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(9, 13)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_y = self.calculate_com_y(plank_vals, plank_index = 2)
        except:
            com_y = 0.5 # mid point of the bed
        finally:
            return com_y
    
    def get_com_4_y(self, x):
        
        '''
        Function calculates y co-ordinate of COM of plank 4.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        sensors_pct_list = ['LC' + str(x) + '_pct' for x in range(13, 17)]
        try:
            plank_vals = x[sensors_pct_list].values
            com_y = self.calculate_com_y(plank_vals, plank_index = 3)
        except:
            com_y = 0.5 # mid point of the bed
        finally:
            return com_y

    def calculate_com_y(self, x, plank_index):
        
        '''
        Function takes the values of a particular and 
        calculates the y co-ordinate of COM for that particular plank.
        '''        
        
        # first left sensor value of plank
        left_val_1 = x[1]
        # second left sensor value of plank
        left_val_2 = x[2]
        # first right sensor value of plank
        right_val_1 = x[0]
        # second right sensor value of plank
        right_val_2 = x[3]
        # total weight percentage of the plank
        total_weight = np.sum(x)
        # for calculating the y cordinate of centre of mass of plank
        com_plank_y = ((right_val_1 *  0) + (left_val_1 * self.plank_dict[plank_index]['size'][1]) + \
            (right_val_2 * 0) + (left_val_2 * self.plank_dict[plank_index]['size'][1]))/total_weight
        normalized_y = com_plank_y/self.plank_dict[plank_index]['size'][1]

        return np.round(normalized_y, 2)
    
    def get_errors_from_fitted_line(self, x):
        
        '''
        Function calculates the errors from fitted line passing through COMs of all planks.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        com_x_list = ['plank_1_com_x', 'plank_2_com_x', 'plank_3_com_x', 'plank_4_com_x']
        com_y_list = ['plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y']
        try:
            com_x_values = x[com_x_list].values # x co-ordinates of coms of planks
            com_y_values = x[com_y_list].values # y co-ordinates of coms of planks
            com_x_for_sample = com_x_values.reshape(-1, 1) # x co-ordinates of centre of mass of a sample
            lr.fit(com_x_for_sample, com_y_values) # fitted line through the coms of all planks
            y_preds = lr.predict(com_x_for_sample) # predicted y co-ordinates of coms of all planks
            abs_errors = np.sum([abs(com_y_values[i] - y_preds[i]) for i in range(len(com_y_values))]) # error in y co-ordinates
        except:
            abs_errors = 0 # return the error as 0
        finally:
            return abs_errors

    def get_deviation_plank_3(self, x):
        
        '''
        Function calculates the deviation of y co-ordinate of COM of plank 3.
        from the fitted line passing through COMs of first two planks.
        Input (x) refers to a sample with its corresponding values.
        '''
        
        com_x_list = ['plank_1_com_x', 'plank_2_com_x', 'plank_3_com_x', 'plank_4_com_x']
        com_y_list = ['plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y']
        try:
        	com_x_values = x[com_x_list].values # x co-ordinates of coms of planks
        	com_y_values = x[com_y_list].values # y co-ordinates of coms of planks
        	lr.fit(com_x_values[:2].reshape(-1, 1), com_y_values[:2]) # fitted line through the coms of planks 1 and plank 2
        	y_preds = lr.predict(com_x_values[2:].reshape(-1, 1)) # predicted y co-ordinates of coms of planks 3 and 4
        	y_preds = list(com_y_values[:2]) + list(y_preds)  # predicted y co-ordinates of coms of all planks
        	plank_3_error = com_y_values[2] - y_preds[2] # deviation of y co-ordinate of plank 3 from the fitted line
        except:
            plank_3_error = 0
        finally:
            return plank_3_error

    def get_deviation_plank_4(self, x):
        
        '''
        Function calculates the deviation of y co-ordinate of COM of plank 4
        from the fitted line passing through COMs of first two planks.
        Input (x) refers to a sample with its corresponding values
        '''
        
        com_x_list = ['plank_1_com_x', 'plank_2_com_x', 'plank_3_com_x', 'plank_4_com_x']
        com_y_list = ['plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y']
        try:
        	com_x_values = x[com_x_list].values # x co-ordinates of coms of planks
        	com_y_values = x[com_y_list].values # y co-ordinates of coms of planks
        	lr.fit(com_x_values[:2].reshape(-1, 1), com_y_values[:2]) # fitted line through the coms of planks 1 and plank 2
        	y_preds = lr.predict(com_x_values[2:].reshape(-1, 1)) # predicted y co-ordinates of coms of planks 3 and 4
        	y_preds = list(com_y_values[:2]) + list(y_preds) # predicted y co-ordinates of coms of all planks
        	#plank_3_error = y_3 - y_preds[2]
        	plank_4_error = com_y_values[3] - y_preds[3] # deviation of y co-ordinate of plank 4 from the fitted line
        except:
            plank_4_error = 0
        finally:
            return plank_4_error

    def bucketize_plank_dev(self, x):
        
        '''
        Function categorises the error value of plank 3 and plank 4 deviation from the fitted line 
        passing through the COMs of first two planks
        '''
        
        dev = 'on'
        threshold = 0.075
        try:
        	#if x >= threshold and x < threshold * 2:
        	#    dev = 'positive_high'
        	#elif x >= threshold * 2:
        	#	dev = 'positive_high_high'
        	#elif x <= -threshold and x > -threshold * 2:
        	#    dev = 'negative_high'
        	#elif x < - threshold * 2:
        	#	dev = 'negative_high_high'
        	if x >= threshold:
        	    dev = 'positive_high'
        	elif x <= -threshold:
        	    dev = 'negative_high'
        except:
            pass
        finally:
            return dev

    def plank_4_wrt_3_2(self, x):

        '''
        Function calculates error of COM of plank 4 from fitted line passing through planks 2 and 3
        '''
            
        com_x_list = ['plank_1_com_x', 'plank_2_com_x', 'plank_3_com_x', 'plank_4_com_x']
        com_y_list = ['plank_1_com_y', 'plank_2_com_y', 'plank_3_com_y', 'plank_4_com_y']
        try:
            com_x_values = x[com_x_list].values # x co-ordinates of coms of planks
            com_y_values = x[com_y_list].values # y co-ordinates of coms of planks
            lr.fit(com_x_values[1:3].reshape(-1, 1), com_y_values[1:3]) # fitted line through the coms of planks 2 and plank 3
            y_pred = lr.predict(com_x_values[3:].reshape(-1, 1)) # predicted y co-ordinates of coms of plank 4
            plank_4_dev_3_2 = y_pred[0] - com_y_values[3] # deviation of y co-ordinate of plank 4 from the fitted line
        except:
            plank_4_dev_3_2 = 0
        finally:
            return plank_4_dev_3_2






