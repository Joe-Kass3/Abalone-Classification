"""
ML - Final Project 

Abalone
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm


class NB:
    
    def __init__(self, data):
        """
        Parameters
        ----------
        data : pd dataframe of features and classes. class is last row

        """
        self.data = data
        
    def describe(self):
        """
        Returns
        -------
        Description of data in pd

        """
        return(self.data.describe())
        
    def correlation(self, pred_feature):
        """
        Parameters
        ----------
        pred_feature : Feature of data that is to be predicted
        
        Returns
        -------
        matrix_corr : np Matrix of correlation of data, rounded to 3 decimals

        """
        matrix_corr = np.round(pd.DataFrame(self.data.iloc[:,:pred_feature].corr()).values, decimals = 3)
        return(matrix_corr)
    
    def seperate_train_test(self, p_train_size, rng=np.random.RandomState()):
        """
        Parameters
        ----------
        p_train_size : % of data that will be used as training set.
        
        Returns
        -------
        train_data : pd dataframe of training data
        test_data : pd dataframe of testing data

        """
        # train_size = int(self.data.size * p_train_size)
        # print(self.data.size > train_size)
        # # print(train_size)
        
        self.train_data = self.data.sample(frac = p_train_size, random_state = rng)
        self.test_data = self.data.drop(self.train_data.index)
        
        return (self.train_data, self.test_data)
    
    def cat_helper(self, rings, bound_1, bound_2):
        """
        Apply across dataframe to sort by age

        """
        if rings <= bound_1:
            return 0
        elif (bound_1 < rings <= bound_2):
            return 1
        elif (rings > bound_2):
            return 2
    
    def age_to_category(self):
        """
        Returns
        -------
        None. Adds column to data with categorization for age

        """
        std = self.data['Rings'].std()
        mean = self.data['Rings'].mean()
        
        bound_1 = norm.ppf(0.33, loc = mean, scale = std)
        bound_2 = norm.ppf(0.66, loc = mean, scale = std)
        
        self.data['Category Age'] = self.data['Rings'].apply(lambda x: self.cat_helper(x, bound_1, bound_2))

    def calc_prior(self):
        """
        Returns
        -------
        prior : prior likelihood of belonging to one fo the 3 classes

        """
        total_num = len(self.train_data)

        self.prior = {0: 0, 1: 0, 2: 0}
    
        count_of_age = self.train_data["Category Age"].value_counts()
        
        self.prior[0] = round((count_of_age[0] / total_num), 3) 
        self.prior[1] = round((count_of_age[1] / total_num), 3) 
        self.prior[2] = round((count_of_age[2] / total_num), 3)
        return self.prior
    
    def calc_likelihood(self):
        pass
    
    
    def train(self, num_iter, bool_remove_feature=False, num_remove_feature=0):
        pass
        
    def classify(self):
        pass


"""Notes:
      
        Assuming age is distributed normally, breaking categorization into 3 equal groups (young, medium, old) is ideal to categorize
        
        Young (<33%)            <= ~8.52 
        Medium (33%< x <66%)      ~8.52 < x <= ~11.3
        Old (>66%)              > 11.3 

"""

if __name__ == "__main__":
    #read in data
    wd = Path().absolute()
    d_path = str(wd) + '\\data\\abalone.data'
    
    data = pd.read_csv(d_path, sep=',', names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'])
    # print(len(data))
    data = pd.get_dummies(data) #convert sex to numerical value
    
    
    nb_abalone = NB(data)
    # print(nb_abalone.data.columns) #need to add column names
    nb_abalone.age_to_category()
    
    
    #process data for meaning
    # nb_abalone.describe()
    
    # print(nb_abalone.correlation(8))
    
    train,test = nb_abalone.seperate_train_test(0.85, 42)
    nb_abalone.calc_prior()
    #print(nb_abalone.calc_prior(train))

    #print/save results