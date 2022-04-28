"""
ML - Final Project 
Abalone
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import math as math

wd = Path().absolute()
pd.set_option('display.max_columns', None)
d_path = str(wd) + "/data/abalone.data"

data = pd.read_csv(d_path, sep=',', names=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings'])

# dataset = [[3.393533211,2.331273381,0],
	# [3.110073483,1.781539638,0],
	# [1.343808831,3.368360954,0],
	# [3.582294042,4.67917911,0],
	# [2.280362439,2.866990263,0],
	# [7.423436942,4.696522875,1],
	# [5.745051997,3.533989803,1],
	# [9.172168622,2.511101045,1],
	# [7.792783481,3.424088941,1],
	# [7.939820817,0.791637231,1]]


def seperate_train_test(data, p_train_size, rng=np.random.RandomState()):
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
    
    train_data = data.sample(frac = p_train_size, random_state = rng)
    test_data = data.drop(train_data.index)
    
    return (train_data, test_data)

train_data, test_data = seperate_train_test(data, .1, 42)

def cat_helper(data, rings, bound_1, bound_2):
	"""
	Apply across dataframe to sort by age
	"""
	if rings <= bound_1:
	    return 0
	elif (bound_1 < rings <= bound_2):
	    return 1
	elif (rings > bound_2):
	    return 2

def age_to_category(data):
    """
    Returns
    -------
    None. Adds column to data with categorization for age
    """
    std = data['Rings'].std()
    mean = data['Rings'].mean()

    bound_1 = norm.ppf(0.33, loc = mean, scale = std)
    bound_2 = norm.ppf(0.66, loc = mean, scale = std)

    data['Category Age'] = data['Rings'].apply(lambda x: cat_helper(data, x, bound_1, bound_2))


age_to_category(train_data)

def separate_class(data):
	separated = dict()

	cat_list = [v for k, v in data.groupby('Category Age')]
	for i in range(len(cat_list)):
		class_value = cat_list[i].iloc[10, 9]
		if class_value not in separated:
			separated[class_value] = cat_list[i]


	return separated


separated = separate_class(train_data)


# for label in separated:
# 	print("~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
# 	print(label)
# 	print(separated[label])


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(data):
	# print(type(data))
	summaries = []

	columns = list(data.columns)
	columns = columns[1:8]

	for column in columns:
		mean = data[column].mean()
		std = data[column].std()
		col_summary = (column, mean, std)
		summaries.append(col_summary)
	return summaries

def summarize_by_class(data):
	separated = separate_class(data)
	# print(separated[0])
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = (summarize_dataset(rows))

	return (summaries)

# print(summarize_by_class(train_data))


def calculate_probability(num, mean, std):
	exponent = math.exp(-((num-mean)**2 / (2 * std**2 )))
	return (1 / (math.sqrt(2 * 3.14) * std)) * exponent

# summaries = summarize_by_class(train_data)
# mean = (summaries[0][1][1])
# std = summaries[0][1][2]
# row = train_data.iloc[0, :]
# print(row[2])

# print(calculate_probability(row[2], mean, std))



def calc_class_probabilities(data, summaries, row):
	total_num = len(data)
	prior_prob = {0:0, 1:0, 2:0}
	probabilities = dict()
	count_of_age = data["Category Age"].value_counts()
	

	prior_prob[0] = round((count_of_age[0] / total_num), 3)
	prior_prob[1] = round((count_of_age[1] / total_num), 3)
	prior_prob[2] = round((count_of_age[2] / total_num), 3)

	for class_values, summaries in summaries.items():
		for i in range(len(summaries)):
			attribute, mean, std = (summaries[i])
			probabilities[class_values] = prior_prob[class_values] * calculate_probability(row[i+1], mean, std)
	print(probabilities)

summaries = summarize_by_class(train_data)
# print(summaries)
calc_class_probabilities(train_data, summaries, train_data.iloc[0, :])


print(train_data)

for i in range(len(train_data)):
	print(i) 
	print(train_data.iloc[i, :])
	print(calc_class_probabilities(train_data, summaries, train_data.iloc[i, :]))
