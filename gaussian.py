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


def separate_class(data):
	"""
	Returns a dictionary that the dataset values separated into the three age classes
	"""
	separated = dict()

	cat_list = [v for k, v in data.groupby('Category Age')]
	for i in range(len(cat_list)):
		class_value = cat_list[i].iloc[10, 9]
		if class_value not in separated:
			separated[class_value] = cat_list[i]


	return separated


# for label in separated:
# 	print("~~~~~~~~~~~~~~~~~~~~~~\n\n\n")
# 	print(label)
# 	print(separated[label])


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(data):
	"""
	Returns a list of mean and standard dev for each column in whatever dataset is passed through
	"""
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
	"""
	Takes the dict separated by age class 
	Creates a dictionary that contains the mean and standard dev for each column of the dataset
	Calculates and returns these values separated by age category
	"""
	separated = separate_class(data)
	# print(separated[0])
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = (summarize_dataset(rows))

	return (summaries)

# print(summarize_by_class(train_data))


def calculate_probability(num, mean, std):
	"""
	Calculating probability using gaussian distribution
	1 / (sqrt(2 * 3.14) * std)) * exp(-((num-mean)**2 / (2 * std**2 ))
	"""
	exponent = math.exp(-((num-mean)**2 / (2 * std**2 )))
	pdf = (1 / (math.sqrt(2 * 3.14) * std)) * exponent
	return pdf


def calc_class_probabilities(data, summaries, row):
	"""
	Calculates the probability for each age category using the summaries provided by previous functions
	"""
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
	
	return probabilities


def prediction(train_data, model, row):
	"""
	Takes in training data and uses that to calculate the probability for each row in the new test data
	"""
	probabilities = calc_class_probabilities(train_data, model, row)
	# print(probabilities.items())
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	
	return best_label


## Running Test ##

train_data, test_data = seperate_train_test(data, .1, 42)
age_to_category(train_data)
age_to_category(test_data)
separated = separate_class(train_data)
model = summarize_by_class(train_data)


# print(len(test_data))
# print(len(train_data))
right = 0
wrong = 0
for i in range(len(test_data)):
	category = test_data.iloc[i, :]["Category Age"]
	label = prediction(train_data, model, test_data.iloc[i, :])
	print("\n~~~\n")
	print("Category:", category)
	print("Best label:", label)
	if category != label:
		print("Wrong!")
		wrong += 1
	elif category == label:
		print("Right!")
		right += 1

print("Total:", right+wrong, "\nRight:", right/(right+wrong), "\nWrong:", wrong/(right+wrong))

#