from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    "max_depth": [3],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1,2 ]
}


def train(X_train, Y_train):
    model = DecisionTreeClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    best_model = grid_search.best_estimator_
    return best_model