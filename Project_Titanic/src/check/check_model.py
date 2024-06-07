from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

def check(best_model, X_test, Y_test):
    Y_pred = best_model.predict(X_test)
    print("Accuracy score: ", accuracy_score(Y_test, Y_pred))
    print("Precision score: ", precision_score(Y_test, Y_pred))
    print("Recall score:", recall_score(Y_test, Y_pred))
    print("Confusion Matrix: ")
    print(confusion_matrix(Y_test, Y_pred))








