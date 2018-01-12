#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#Imports
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *
import warnings
warnings.filterwarnings('ignore')
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

#Exploring the Dataset
print "\nExploring the Dataset\n====================="
print 'Number of features: ', len(features_list)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


total_number_people = len(data_dict)
num_poi = 0
for i in data_dict:
    if data_dict[i]['poi']==True:
        num_poi=num_poi+1

print "Number of datapoint:", len(data_dict)
print 'Number of poi:', num_poi
print 'Number of non poi:', len(data_dict)-num_poi


### Task 2: Remove outliers

# Visualize the outlier
features =['salary', 'bonus']
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show() 

for key, val in data_dict.items():
    if val['salary'] != 'NaN' and val['salary'] > 10000000:
        print key
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
print "\n=========>Removing the outliers<========="
print "Number of datapoint excluding outliers:", len(data_dict)

# Visualize without the outlier
data = featureFormat(data_dict, features)
for point in data:
    salary=point[0]
    bonus=point[1]
    plt.scatter(salary, bonus)
plt.xlabel('salary')
plt.ylabel('bonus')
plt.show()

### Store to my_dataset for easy export below.
my_dataset = data_dict
                    
### Missing values in each feature
nan = [0 for i in range(len(features_list))]
for i, person in my_dataset.iteritems():
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            nan[j] += 1
print "\nNAN count\n{}".format("="*len("NAN count"))
for i, feature in enumerate(features_list):
    print feature,':', nan[i]

### Task 3: Create new feature(s)
# Bonus-salary ratio
for employee, features in data_dict.iteritems():
	if features['bonus'] == "NaN" or features['salary'] == "NaN":
		features['bonus_salary_ratio'] = "NaN"
	else:
		features['bonus_salary_ratio'] = float(features['bonus']) / float(features['salary'])

# from_this_person_to_poi as a percentage of from_messages
for employee, features in data_dict.iteritems():
	if features['from_this_person_to_poi'] == "NaN" or features['from_messages'] == "NaN":
		features['from_this_person_to_poi_percentage'] = "NaN"
	else:
		features['from_this_person_to_poi_percentage'] = float(features['from_this_person_to_poi']) / float(features['from_messages'])

# from_poi_to_this_person as a percentage of to_messages
for employee, features in data_dict.iteritems():
	if features['from_poi_to_this_person'] == "NaN" or features['to_messages'] == "NaN":
		features['from_poi_to_this_person_percentage'] = "NaN"
	else:
		features['from_poi_to_this_person_percentage'] = float(features['from_poi_to_this_person']) / float(features['to_messages'])

### Impute missing email features to mean
email_features = ['to_messages',
	              'from_poi_to_this_person',
	              'from_poi_to_this_person_percentage',
	              'from_messages',
	              'from_this_person_to_poi',
	              'from_this_person_to_poi_percentage',
	              'shared_receipt_with_poi']
 
email_feature_sums = defaultdict(lambda:0)
email_feature_counts = defaultdict(lambda:0)

for employee, features in data_dict.iteritems():
	for ef in email_features:
		if features[ef] != "NaN":
			email_feature_sums[ef] += features[ef]
			email_feature_counts[ef] += 1

email_feature_means = {}
for ef in email_features:
	email_feature_means[ef] = float(email_feature_sums[ef]) / float(email_feature_counts[ef])

for employee, features in data_dict.iteritems():
	for ef in email_features:
		if features[ef] == "NaN":
			features[ef] = email_feature_means[ef]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Potential pipeline steps
def pipe(i):
	scaler = MinMaxScaler()
	select = SelectKBest()
	dtc = DecisionTreeClassifier()
	svc = SVC()
	knn = KNeighborsClassifier()


	if i==0:
		print "\n\nDecision Tree Classifier\n{}".format("="*len("Decision Tree Classifier"))
		steps = [# Preprocessing
		         ('min_max_scaler', scaler),
		    
		         # Feature selection
		         ('feature_selection', select),
		         
		         # Classifier
		         ('dtc', dtc)
		         ]

		# Parameters to try in grid search
		parameters = dict(
		                  feature_selection__k=[2, 3, 5, 6],
		                  dtc__criterion=['gini', 'entropy'],
		                  dtc__max_depth=[None, 1, 2, 3, 4],
		                  dtc__min_samples_split=[1, 2, 3, 4, 25],
		                  dtc__class_weight=[None, 'balanced'],
		                  dtc__random_state=[42]
		                  )
	elif i==1:
		print "\n\n K-Nearest Neighbor\n{}".format("="*len(" K-Nearest Neighbor"))
		steps = [# Preprocessing
		         ('min_max_scaler', scaler),
		    
		         # Feature selection
		         ('feature_selection', select),
		         
		         # Classifier
		         ('knn', knn)
		         ]

		# Parameters to try in grid search
		parameters = dict(
		                  feature_selection__k=[2, 3, 5, 6],
                  		  knn__n_neighbors=[1, 2, 3, 4, 5],
                  		  knn__leaf_size=[1, 10, 30, 60],
                  		  knn__algorithm=['auto', 'ball_tree', 'kd_tree', 'brute']
		                 )

	else:
		print "\n\nSupport Vector Machine Classifier\n{}".format("="*len("Support Vector Machine Classifier"))
		steps = [# Preprocessing
		         ('min_max_scaler', scaler),
		    
		         # Feature selection
		         ('feature_selection', select),
		         
		         # Classifier
		         ('svc', svc)
		         ]

		# Parameters to try in grid search
		parameters = dict(
		                  feature_selection__k=[2, 3, 5, 6],
		                  svc__C=[0.1, 1, 10, 100, 1000],
		                  svc__kernel=['rbf'],
		                  svc__gamma=[0.001, 0.0001],
		                  )

	# Create pipeline
	pipeline = Pipeline(steps)
	return (pipeline, parameters)
	
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# Train-Test Split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


# Provided to give you a starting point. Try a variety of classifiers.
def reporting(labels_test, labels_predictions, clf):
	report = classification_report( labels_test, labels_predictions )
	print(report)

	### Task 6: Dump your classifier, dataset, and features_list so anyone can
	### check your results. You do not need to change anything below, but make sure
	### that the version of poi_id.py that you submit can be run on its own and
	### generates the necessary .pkl files for validating your results.

	dump_classifier_and_data(clf, my_dataset, features_list)

# Gaussian Naive Bayes
print "\n\nGaussian Naive Bayes\n{}".format("="*len("Gaussian Naive Bayes"))
gnb = GaussianNB()
gnb.fit(features_train,labels_train)
pred = gnb.predict(features_test)
acc = accuracy_score(pred,labels_test)

print "GaussianNB Accuracy:", acc
reporting(labels_test,pred,gnb)


def GSCV(i,features_train, features_test, labels_train, labels_test):
	# Cross-validation for parameter tuning in grid search 
	sss = StratifiedShuffleSplit(
	    labels_train,
	    n_iter = 20,
	    test_size = 0.5,
	    random_state = 0
	    )

	pipeline, parameters = pipe(i)
	# Create, fit, and make predictions with grid search
	gs = GridSearchCV(pipeline,
		              param_grid=parameters,
		              scoring="f1",
		              cv=sss,
		              error_score=0)
	gs.fit(features_train, labels_train)
	labels_predictions = gs.predict(features_test)

	# Pick the classifier with the best tuned parameters
	clf = gs.best_estimator_
	print "\n", "Best parameters are: ", gs.best_params_, "\n"

	features_selected=[features_list[i+1] for i in clf.named_steps['feature_selection'].get_support(indices=True)]
	scores = clf.named_steps['feature_selection'].scores_
	return (labels_test, labels_predictions, clf)

# To see Desision Tree (0), K-Nearest Neighbor (1), SVC(2) => uncomment below lines 
"""
model = [0,1,2]
for i in model:
	labels_test, labels_predictions, clf= GSCV(i,features_train, features_test, labels_train, labels_test)
	reporting(labels_test, labels_predictions, clf)
	"""