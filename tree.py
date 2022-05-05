"""
Decision Tree - Final Project 
Abalone
"""

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import time


def plot_predicted(real_data, predicted_data):
    """
    Parameters
    ----------
    real_target : actual values of predicted target
    predicted_target : values predicted by classifier

    Returns
    -------
    Root mean square error

    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(real_data, predicted_data, '.k')
    
    ax.plot([0, 30], [0, 30], '--k')
    ax.plot([0, 30], [2, 32], ':k')
    ax.plot([2, 32], [0, 30], ':k')
    
    rms = (real_data - predicted_data).std()
    
    ax.text(25, 3,
            "Root Mean Square Error = %.2g" % rms,
            ha='right', va='bottom')

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    
    ax.set_xlabel('Real number of rings')
    ax.set_ylabel('Predicted number of rings')
    
    return rms


def iter_max_depth(n, m, step_size, feat_train, feat_test, tar_train, tar_test):
    """
    Parameters
    ----------
    n : starting number to train max depth
    m : ending number to train max depth
    feature and target data : data split by train_test_split

    Returns
    -------
    rms_test : error in the testing set
    rms_train : error in training set

    """
    rms_test = []
    rms_train = []
    
    for i in range(n, m, step_size):
        model = DecisionTreeRegressor(max_depth=i)
    
        model.fit(feat_train, tar_train)
    
        predicted_test = model.predict(feat_test)
        predicted_train = model.predict(feat_train)

        rms_test.append(plot_predicted(tar_train, predicted_train))
        plt.title("Training data -  Max Depth: %s" % i)
        rms_train.append(plot_predicted(tar_test, predicted_test))
        plt.title("Test data -  Max Depth: %s" % i)
        plt.clf() 
        
    return(rms_test, rms_train)

def iter_random_forest(n, m, feat_train, feat_test, tar_train, tar_test):
    """
    Parameters
    ----------
    n : min number of estimators to train
    m : max number of estimators to train
    feature and target data : data split by train_test_split

    Returns
    -------
    rms_test : error in the testing set
    rms_train : error in training set

    """
    rms_test = []
    rms_train = []
    
    for i in range(n, m):
        model = RandomForestRegressor(n_estimators=i)
    
        model.fit(feat_train, tar_train)
    
        predicted_test = model.predict(feat_test)
        predicted_train = model.predict(feat_train)

        rms_test.append(plot_predicted(tar_train, predicted_train))
        plt.title("Training data -  Max Depth: %s" % i)
        rms_train.append(plot_predicted(tar_test, predicted_test))
        plt.title("Test data -  Max Depth: %s" % i)
        
        plt.clf() 
        
    return(rms_test, rms_train)

def plot_error(rms_test, rms_train):
    """
    Parameters
    ----------
    rms_test : error in the testing set
    rms_train : error in training set

    """
    
    plt.plot(rms_test, label='Testing') 
    plt.plot(rms_train, label='Training')
    plt.legend(loc=5)
    # plt.title("Increasing Size of Random Forest")
    # # plt.xlabel('Increment of 5 levels')
    plt.ylabel('RMS Error')
    
    
if __name__ == "__main__":
    wd = Path().absolute()
    pd.set_option('display.max_columns', None)
    d_path = str(wd) + "/data/abalone.data"

    data = pd.read_csv(d_path, sep=',', names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'])

    data = pd.get_dummies(data)

    target = data['Rings'].values
    data = data.drop(['Rings'],1)
    
    feature = data.values.astype(float)
    

    feat_train, feat_test, tar_train, tar_test = model_selection.train_test_split(feature, target)
    
    # Below confirms that automatic depth of the decision tree is 26 and number of unique features is 28
    model = DecisionTreeRegressor()
    model.fit(feat_train, tar_train)
    print("Depth of Tree: " + str(model.get_depth()))
    print("Number of leaves: " + str(model.get_n_leaves()))
    print("Number of possible ages: " + str(len(np.unique(target))))
    
    # Below investigates importance of max depth of tree
    # start = time.time()
    rms_test, rms_train = iter_max_depth(5, 106, 5, feat_train, feat_test, tar_train, tar_test)
    # end = time.time()
    # print(end - start)
    plot_error(rms_test, rms_train)
    
    # Below investigates importance of using a random forest ensemble
    rms_test, rms_train = iter_random_forest(10, 50, feat_train, feat_test, tar_train, tar_test)
    plot_error(rms_test, rms_train)
    