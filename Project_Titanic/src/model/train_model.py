import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import svm

directory_scores = "scores"
def train_decision_tree(X_train, Y_train):

    param_grid = {
        "max_depth": [3],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_

    #Added save plots to driectory
    directory_scores_target = "Decision_Tree_scores"
    file_name = "decision_tree_plot.png"
    file_path = os.path.join(directory_scores,directory_scores_target, file_name)
    plt.figure(figsize=(10, 6))
    plot_tree(best_model, filled=True, feature_names=X_train.columns, class_names=["Class 0", "Class 1"])
    plt.savefig(file_path)

    return best_model

def train_svm(X_train, Y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, Y_train)
    return clf


def train_random_forest(X_train, Y_train):

    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
       # 'min_samples_split': [2],
      #  'min_samples_leaf': [1]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1')
    grid_search.fit(X_train, Y_train)
    improved_rf = grid_search.best_estimator_

    all_trees = improved_rf.estimators_

    # Added ten trees
    directory_scores_target = "Random_Forest_scores"
    for i, tree in enumerate(all_trees[:10]):
        plt.figure(figsize=(20, 10))
        plot_tree(tree, feature_names=X_train.columns, filled=True)
        file_name = f"decision_tree_{i+1}.png"
        file_path = os.path.join(directory_scores, directory_scores_target, file_name)
        plt.savefig(file_path)

    return improved_rf