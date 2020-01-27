import os
import pickle
import glob

class Classifier(object):
    
    '''
    Class contains function to classify the test data using the latest trained model. 
    '''
    
    def __init__(self, model_folder):
        self.model_folder = model_folder  # folder for models          
        self.model_name = max(glob.glob(model_folder + '/*.sav'), key=os.path.getctime)  # name of the latest model
        
    def classify_model(self, X_test):
        
        '''
        Function to classify test data using the latest trained model
        '''
        
        loaded_model = pickle.load(open(os.path.join(self.model_folder, self.model_name), 'rb'))
        predictions = loaded_model.predict(X_test)
        pred_prob = loaded_model.predict_proba(X_test)
        return predictions, pred_prob
