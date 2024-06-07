from sklearn.metrics import accuracy_score

def check(best_model, X_test, Y_test):
    Y_pred = best_model.predict(X_test)
    print(accuracy_score(Y_test, Y_pred))
