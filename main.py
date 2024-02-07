# Imports
import pickle
import pandas as pd
import numpy as np
import scipy
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Function to read csv and return a dataframe object
def readcsv(filename):

    # Read csv
    data_df = pd.read_csv(filename)

    return data_df

# Function to count each class occurence in a dataframe
def eval_data(data_frame):

    df_count = data_frame['duration_label'].value_counts()

    return df_count

# Function to apply k-fold cross validation on training data
def cross_validation(train_df, k):

    kf = KFold(n_splits=k)

    k_fold_split = []
    for train_index, test_index in kf.split(train_df):
        k_fold_split.append([train_index, test_index])

    return k_fold_split

# Function for 0R model
def zero_r(train_split, test_split, flag):

    # Find the most common duration label
    common_label = train_split['duration_label'].value_counts().idxmax()

    # Not for Kaggle
    if (flag == 0):
        y_test_split = test_split.loc[:, 'duration_label']

        # Predict with most common label
        y_test_predicted = [common_label] * len(test_split)

        return accuracy_score(y_test_split, y_test_predicted)

    else:
        # Predict with most common label
        y_test_predicted = [common_label] * len(test_split)

        return y_test_predicted

# Function for Decision Tree model
def decision_tree(train_split, test_split, flag):
    
    # Feature select number of steps and ingredients 
    X_train = train_split.loc[:, ['n_steps','n_ingredients']]
    y_train = train_split.loc[:, 'duration_label']

    X_test = test_split.loc[:, ['n_steps','n_ingredients']]
    
    # Fit training set to decision tree
    model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3)
    clf = model.fit(X_train, y_train)

    # Distinguish from normal prediction and kaggle
    if flag == 0:
        y_test = test_split.loc[:, 'duration_label'] 

        # Calculate accuracy score for the model
        score_te = model.score(X_test, y_test)

        # # Tree summary and model evaluation metrics for decision tree
        # print("--------------------- Tree Summary ---------------------")
        # print('Classes: ', clf.classes_)
        # print('Tree Depth: ', clf.tree_.max_depth)
        # print('No. of leaves: ', clf.tree_.n_leaves)
        # print('No. of features: ', clf.n_features_)
        # print("--------------------------------------------------------\n")

        # Uncomment to visualise decision tree model
        # fig = plt.figure(figsize=(100,90))
        # _ = tree.plot_tree(clf, 
        #            feature_names=['n_steps','n_ingredients'],  
        #            class_names=["1.0","2.0","3.0"],
        #            filled=True)

        # fig.savefig("decision_tree.png")

        return score_te

    else:
        # Make class predictions based on test features
        prediction = clf.predict(X_test)

        return prediction

# Function to create a decision tree based on engineered features
def decision_tree_combine(train_split, test_split, flag):

    # Feature select number of steps and ingredients 
    X_train = train_split.loc[:, ['n_steps','n_ingredients']]
    y_train = train_split.loc[:, 'duration_label']

    # Feature engineer new feature n_steps * n_ingredients
    X_train_new = pd.DataFrame(X_train['n_steps']*X_train['n_ingredients'])
    X_test = test_split.loc[:, ['n_steps','n_ingredients']]
    X_test_new = pd.DataFrame(X_test['n_steps']*X_test['n_ingredients'])
    
    # Fit training set to decision tree
    model = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3)
    clf = model.fit(X_train_new, y_train)

    # Distinguish from normal prediction and kaggle
    if flag == 0:
        y_test = test_split.loc[:, 'duration_label'] 

        # Calculate accuracy score for the model
        score_te = model.score(X_test_new, y_test)

        # # Tree summary and model evaluation metrics
        # print("--------------------- Tree Summary ---------------------")
        # print('Classes: ', clf.classes_)
        # print('Tree Depth: ', clf.tree_.max_depth)
        # print('No. of leaves: ', clf.tree_.n_leaves)
        # print('No. of features: ', clf.n_features_)
        # print("--------------------------------------------------------\n")

        # fig = plt.figure(figsize=(100,90))
        # _ = tree.plot_tree(clf, 
        #            feature_names=['n_steps','n_ingredients'],  
        #            class_names=["1.0","2.0","3.0"],
        #            filled=True)

        # fig.savefig("decision_tree_comb.png")

        return score_te

    else:
        # Make class predictions based on test features
        prediction = clf.predict(X_test_new)

        return prediction

# Function for KNN Classifier
def knn_classifier(filename, feature_name, train_split, test_split, k, flag):

    vocab = pickle.load(open(filename, "rb"))

    # Select training data
    X_train_feature = train_split.loc[:, feature_name]
    y_train_feature = train_split.loc[:, 'duration_label']

    # Get testing data
    X_test_feature = test_split.loc[:, feature_name]

    # Fit count vectorisor to training model as training corpus
    training_data = X_train_feature.values
    X_train = vocab.fit_transform(training_data)
    X_train_array = np.copy(X_train.toarray())

    # Get count vectorisation for X_test based on X_train corpus
    testing_data = X_test_feature.values
    X_test = vocab.transform(testing_data)
    X_test_array = np.copy(X_test.toarray())

    # Using euclidean distance p = 2
    model = KNeighborsClassifier(n_neighbors = k, p = 2, metric='minkowski')

    # Fit model using training set
    model.fit(X_train_array, y_train_feature)

    # Distinguish from normal prediction and kaggle
    if flag == 0:
        y_test_feature = test_split.loc[:, 'duration_label']

        # Predict with test set
        accuracy = model.score(X_test_array, y_test_feature)

        return accuracy
    
    else:
        prediction = model.predict(X_test)

        return prediction

def main(train_file, test_file):

    # Train file to df
    train_df = readcsv(train_file)

    # Test file to df (for Kaggle)
    test_df = readcsv(test_file)

    # Evaluate dataset
    train_df_count = eval_data(train_df)
    # Each class count
    print("Occurrences of each class:\n", train_df_count)

    # Zero R
    acc_zero_r = []

    # Decision Tree
    acc_dt = []
    acc_dt_comb = []

    # KNN Classifier
    acc_name_3nn = []
    acc_name_5nn = []
    acc_name_7nn = []
    acc_steps_3nn = []
    acc_steps_5nn = []
    acc_steps_7nn = []
    acc_ingr_3nn = []
    acc_ingr_5nn = []
    acc_ingr_7nn = []

    # Cross Validation on train_file, k = 5
    index_split = cross_validation(train_df, 5)

    # Perform model fit and prediction on each cross validation set
    for each_split in index_split:
        train_index_split = each_split[0]
        test_index_split = each_split[1]

        # Select training data
        train_split = train_df.iloc[train_index_split]
        test_split = train_df.iloc[test_index_split]

        # ----------- Call each classifier -----------

        # Zero R
        each_zero_r = zero_r(train_split, test_split, 0)
        acc_zero_r.append(each_zero_r)

        # Decision Tree using Gini
        each_decision_tree = decision_tree(train_split, test_split, 0)
        acc_dt.append(each_decision_tree)

        # Feature Engineered Decision Tree
        each_decision_tree_comb = decision_tree_combine(train_split, test_split, 0)
        acc_dt_comb.append(each_decision_tree_comb)

        # -------------------------- Comment some K-nn out to improve run time --------------------------
        # KNN with k = 3,5,7 for Name (CountVectorizer)
        each_3nn_name = knn_classifier("train_name_countvectorizer.pkl", "name", train_split, test_split, 3, 0)
        each_5nn_name = knn_classifier("train_name_countvectorizer.pkl", "name", train_split, test_split, 5, 0)
        each_7nn_name = knn_classifier("train_name_countvectorizer.pkl", "name", train_split, test_split, 7, 0)
        acc_name_3nn.append(each_3nn_name)
        acc_name_5nn.append(each_5nn_name)
        acc_name_7nn.append(each_7nn_name)

        # KNN with k = 3,5,7 for Steps (CountVectorizer)
        each_3nn_steps = knn_classifier("train_steps_countvectorizer.pkl", "steps", train_split, test_split, 3, 0)
        each_5nn_steps = knn_classifier("train_steps_countvectorizer.pkl", "steps", train_split, test_split, 5, 0)
        each_7nn_steps = knn_classifier("train_steps_countvectorizer.pkl", "steps", train_split, test_split, 7, 0)
        acc_steps_3nn.append(each_3nn_steps)
        acc_steps_5nn.append(each_5nn_steps)
        acc_steps_7nn.append(each_7nn_steps)

        # KNN with k = 3,5,7 for Ingredients (CountVectorizer)
        each_3nn_ingr = knn_classifier("train_ingr_countvectorizer.pkl", "ingredients", train_split, test_split, 3, 0)
        each_5nn_ingr = knn_classifier("train_ingr_countvectorizer.pkl", "ingredients", train_split, test_split, 5, 0)
        each_7nn_ingr = knn_classifier("train_ingr_countvectorizer.pkl", "ingredients", train_split, test_split, 7, 0)
        acc_ingr_3nn.append(each_3nn_ingr)
        acc_ingr_5nn.append(each_5nn_ingr)
        acc_ingr_7nn.append(each_7nn_ingr)

    # For Kaggle - use recipe_test.csv

    # Zero R
    kaggle_zero_r = zero_r(train_df, test_df, 1)

    with open('kaggle_0R.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "duration_label"])
        id_index = 1
        for prediction in kaggle_zero_r:
            writer.writerow([id_index, prediction])
            id_index+=1

    # Decision Tree
    kaggle_dt = decision_tree(train_df, test_df, 1)

    with open('kaggle_DT.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "duration_label"])
        id_index = 1
        for prediction in kaggle_dt:
            writer.writerow([id_index, prediction])
            id_index+=1

    # 5NN with Steps
    kaggle_5nn_steps = knn_classifier("train_steps_countvectorizer.pkl", "steps", train_df, test_df, 5, 1)

    with open('kaggle_steps_5NN.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "duration_label"])
        id_index = 1
        for prediction in kaggle_5nn_steps:
            writer.writerow([id_index, prediction])
            id_index+=1

    # 7NN with Steps
    kaggle_7nn_steps = knn_classifier("train_steps_countvectorizer.pkl", "steps", train_df, test_df, 7, 1)

    with open('kaggle_steps_7NN.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "duration_label"])
        id_index = 1
        for prediction in kaggle_7nn_steps:
            writer.writerow([id_index, prediction])
            id_index+=1

    return acc_zero_r, acc_dt, acc_dt_comb, acc_name_3nn, acc_name_5nn, acc_name_7nn, acc_steps_3nn, acc_steps_5nn, acc_steps_7nn, acc_ingr_3nn, acc_ingr_5nn, acc_ingr_7nn

# Get values
total_accuracy = main("recipe_train.csv", "recipe_test.csv")
zero_r_acc = total_accuracy[0]
dt_acc = total_accuracy[1]
dt_comb_acc = total_accuracy[2]
name_3nn_acc = total_accuracy[3]
name_5nn_acc = total_accuracy[4]
name_7nn_acc = total_accuracy[5]
steps_3nn_acc = total_accuracy[6]
steps_5nn_acc = total_accuracy[7]
steps_7nn_acc = total_accuracy[8]
ingr_3nn_acc = total_accuracy[9]
ingr_5nn_acc = total_accuracy[10]
ingr_7nn_acc = total_accuracy[11]

# Print results
print("0-R Average Accuracy: ", np.mean(zero_r_acc))
print("Decision Tree Average Accuracy: ", np.mean(dt_acc))
print("Combined Features Decision Tree Average Accuracy: ", np.mean(dt_comb_acc))
print("3-NN with Name Feature Average Accuracy: ", np.mean(name_3nn_acc))
print("5-NN with Name Feature Average Accuracy: ", np.mean(name_5nn_acc))
print("7-NN with Name Feature Average Accuracy: ", np.mean(name_7nn_acc))
print("3-NN with Steps Feature Average Accuracy: ", np.mean(steps_3nn_acc))
print("5-NN with Steps Feature Average Accuracy: ", np.mean(steps_5nn_acc))
print("7-NN with Steps Feature Average Accuracy: ", np.mean(steps_7nn_acc))
print("3-NN with Ingredients Feature Average Accuracy: ", np.mean(ingr_3nn_acc))
print("5-NN with Ingredients Feature Average Accuracy: ", np.mean(ingr_5nn_acc))
print("7-NN with Ingredients Feature Average Accuracy: ", np.mean(ingr_7nn_acc))
