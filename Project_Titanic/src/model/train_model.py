import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import svm


def train_decision_tree(X_train: pd.DataFrame, Y_train: pd.DataFrame, directory_scores: str,
                        decision_tree_params: dict) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=decision_tree_params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    # Added save plots to driectory
    directory_scores_target = "Decision_Tree_scores"
    file_name = "decision_tree_plot.png"
    file_path = os.path.join(directory_scores, directory_scores_target, file_name)
    plt.figure(figsize=(10, 6))
    plot_tree(best_model, filled=True, feature_names=X_train.columns, class_names=["Class 0", "Class 1"])
    plt.savefig(file_path)

    return best_model


def train_svm(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> svm:
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    return clf


def train_random_forest(X_train: pd.DataFrame, Y_train: pd.DataFrame, directory_scores: str,
                        random_forest_parameters: dict) -> RandomForestClassifier:
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid=random_forest_parameters, cv=3, scoring='f1')
    grid_search.fit(X_train, Y_train)
    improved_rf = grid_search.best_estimator_
    all_trees = improved_rf.estimators_

    # Added ten trees
    directory_scores_target = "Random_Forest_scores"
    for i, tree in enumerate(all_trees[:10]):
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=X_train.columns, filled=True)
        file_name = f"decision_tree_{i + 1}.png"
        file_path = os.path.join(directory_scores, directory_scores_target, file_name)
        plt.savefig(file_path)

    return improved_rf
