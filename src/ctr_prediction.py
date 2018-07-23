# -*- coding: utf-8 -*-
import sys
import csv
import time
import pickle
import numpy as np
import warnings

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from output import printGreen, printYellow, printRed

warnings.filterwarnings("ignore") # Some depreciate warnings regarding scikit in online learning

#
# PROCESS DATA
#
def process_data(samples, offset=0):
    # Build dataframes
    X_dict = []
    y = []

    # Open file
    with open('../data/raw/train', 'r') as csvfile:
        # Create reader
        reader = csv.DictReader(csvfile)
        for i in range(offset):
            next(reader)
        i = 0
        for row in reader:
            i += 1

            # Append Label to y
            y.append(int(row['click']))
            # Remove features
            del row['click'], row['id'], row['hour'], row['device_id'], row['device_ip']
            
            # Append input to X
            X_dict.append(row)
            if i >= samples:
                break
    
    return X_dict, y

#
# DECISION TREE ~20 min to train
#
def decision_tree(load_model=False):
    start = time.time()
    if load_model == False:
        printYellow("*  Decision tree model training started...")

    # Create training set of 100,000 samples
    n_max = 100000
    X_dict_train, y_train = process_data(100000)

    # Transform training dictionary into one-hot encoded vectors
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
    # print(len(X_train[0]))

    # Creating test set and turn into one-hot encoded vectors
    X_dict_test, y_test = process_data(100000, 100000)
    X_test = dict_one_hot_encoder.transform(X_dict_test)
    # print(len(X_test[0]))
    
    # Load Model
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        d_tree_file = open('../models/decision_tree_model.sav', 'rb')
        decision_tree_final = pickle.load(d_tree_file)
        # d_tree_file.close()

        # Evaluate model on test set
        prob = decision_tree_final.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, prob)
        printGreen('✔  ROC AUC score on test set: {0:.3f}'.format(score))
        d_tree_file.close()
        return 0

    # Train decision tree classifier
    params = {'max_depth': [3, 10, None]}
    decision_tree_model = DecisionTreeClassifier(criterion='gini',
                                                 min_samples_split=30)
    grid_search = GridSearchCV(decision_tree_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    # print("Training started..")
    grid_search.fit(X_train, y_train)
    printGreen('✔  Decision tree model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use model with best parameter as final model
    decision_tree_final = grid_search.best_estimator_

    # Evaluate and run model on training data
    prob = decision_tree_final.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, prob)
    printGreen('✔  ROC AUC score on test set: {0:.3f}'.format(score))

    # Save Model
    decision_tree_model_file = open('../models/decision_tree_model.sav', "wb")
    pickle.dump(decision_tree_final, decision_tree_model_file)
    decision_tree_model_file.close()
    printGreen('✔  Decision tree model saved...')

    return 0

#
# RANDOM FOREST ~ 20 min to train
#
def random_forest(load_model=False):
    start = time.time()
    if load_model == False:
        printYellow("*  Random forest model training started...")

    # Create training set of 100,000 samples
    n_max = 100000
    X_dict_train, y_train = process_data(100000)

    # Transform training dictionary into one-hot encoded vectors
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

    # Creating test set and turn into one-hot encoded vectors
    X_dict_test, y_test = process_data(100000, 100000)
    X_test = dict_one_hot_encoder.transform(X_dict_test)

    # Load model instead of training again..
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        r_forest_file = open('../models/random_forest_model.sav', 'rb')
        random_forest_final = pickle.load(r_forest_file)
        probs = random_forest_final.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probs)
        printGreen('✔  ROC AUC score on test set: {0:.3f}'.format(score))
        r_forest_file.close()
        return 0
    
    # Train random forest classifier
    params = {'max_depth': [3, 10, None]}
    random_forest_model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30,
                                                 n_jobs=-1)
    grid_search = GridSearchCV(random_forest_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    printGreen('✔  Random forest model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use best paramter for final model
    random_forest_final = grid_search.best_estimator_

    # Evaluate model
    probs = random_forest_final.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, probs)
    printGreen('✔  ROC AUC score on test set: {0:.3f}'.format(score))

    # Save Model
    random_forest_file = open('../models/random_forest_model.sav', "wb")
    pickle.dump(random_forest_final, random_forest_file)
    random_forest_file.close()
    printGreen('✔  Random forest model saved...')
    return 0

#
# SGD-BASED LOGISTIC REGRESSION ~20 sec. to train
#
def logistic_regression(sample_size=100000, load_model=False):
    start = time.time()
    if load_model == False:
        printYellow("*  Logistic regression model training started...")

    # Create Training Set
    n = sample_size
    X_dict_train, y_train = process_data(n)
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

    # Create Test Set
    X_dict_test, y_test = process_data(n, n)
    X_test = dict_one_hot_encoder.transform(X_dict_test)

    X_train_n = X_train
    y_train_n = np.array(y_train)

    # Load model instead of training again
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        l_reg_file = open('../models/logistic_regression_model.sav', 'rb')
        sgd_log_reg_model = pickle.load(l_reg_file)
        predictions = sgd_log_reg_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, predictions)
        printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))
        return 0

    # Create SGD Logistic Regression Classifier
    sgd_log_reg_model = SGDClassifier(loss='log', penalty=None, fit_intercept=True,
                                      n_iter=5, learning_rate='constant', eta0=0.01)

    # Train Classifier
    sgd_log_reg_model.fit(X_train_n, y_train_n)
    printGreen('✔  Logistic regression model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Run model on test set
    predictions = sgd_log_reg_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    score = roc_auc_score(y_test, predictions)
    printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save model
    l_reg_file = open('../models/logistic_regression_model.sav', "wb")
    pickle.dump(sgd_log_reg_model, l_reg_file)
    l_reg_file.close()
    printGreen('✔  Logistic regression model saved...')

#
# LOGISTIC REGRESSION USING ONLINE LEARNING ~6 min. to train
#
def logistic_regression_ol(load_model=False):
    start = time.time()
    if load_model == False:
        printYellow("*  Logistic regression (using online learning) model training started...")

    # Build Classifier
    sgd_log_reg_model = SGDClassifier(loss='log', penalty=None, fit_intercept=True, n_iter=1, learning_rate='constant', eta0=0.01)
    
    # Training sets
    X_dict_train, y_train = process_data(100000)
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
    
    X_train_100k = X_train
    y_train_100k = np.array(y_train)

    # Test sets
    X_dict_test, y_test_next10k = process_data(10000, 100000)
    X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)

    
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        l_reg_file = open('../models/logistic_regression_model_ol.sav', 'rb')
        sgd_log_reg_model = pickle.load(l_reg_file)
        X_dict_test, y_test_next = process_data(10000, (20 + 1) * 200000)
        X_test_next = dict_one_hot_encoder.transform(X_dict_test)
        predict = sgd_log_reg_model.predict_proba(X_test_next)[:, 1]
        score = roc_auc_score(y_test_next, predict)
        printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))
        return 0

    # Train and partially fit on 1 million samples
    for i in range(20):
        X_dict_train, y_train_every = process_data(100000, i * 100000)
        X_train_every = dict_one_hot_encoder.transform(X_dict_train)
        sgd_log_reg_model.partial_fit(X_train_every, y_train_every, classes=[0, 1])
    
    printGreen('✔  Logistic regression (using online learning) model training complete..."\t\t{0:.1f}s'.format(time.time() - start))
    
    # Get test set
    X_dict_test, y_test_next = process_data(10000, (i + 1) * 200000)
    X_test_next = dict_one_hot_encoder.transform(X_dict_test)
    
    # Evaluate
    predict = sgd_log_reg_model.predict_proba(X_test_next)[:, 1]
    score = roc_auc_score(y_test_next, predict)
    printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save Model
    l_reg_file = open('../models/logistic_regression_model_ol.sav', "wb")
    pickle.dump(sgd_log_reg_model, l_reg_file)
    l_reg_file.close()
    printGreen('✔  Logistic regression (using online learning) model saved...')
    return 0

#
# MAIN
#
def main():
    # Initial Message
    printGreen("Click-through rate models training started...\n")

    # Decision Tree
    printGreen('Decision Tree')
    decision_tree(load_model=True)
    print('\n')

    # Random Forest
    printGreen('Random Forest')
    random_forest(load_model=False)
    print('\n')

    # Logistic Regression
    printGreen('SGD Based Logistic Regression')
    logistic_regression(load_model=True)
    print('\n')

    # OL Logistic Regression
    printGreen('Logistic Regressions using Online Learning')
    logistic_regression_ol(load_model=True)
    print('\n')

    printGreen("✔  Done")

if __name__ == '__main__':
    main()