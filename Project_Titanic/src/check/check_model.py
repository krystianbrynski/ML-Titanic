from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import os

def check(model, X_test, Y_test, model_name):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    directory_scores = "scores"

    file_name = "Scores_"+ model_name + ".txt" # model_name = name of classificator and directory target
    directory_target = model_name
    file_path = os.path.join(directory_scores,directory_target, file_name)

    with open(file_path, "w") as file:# save scores to txt
        file.write("Scores for " + model_name + "\n")
        file.write("Accuracy score: " + str(accuracy) + "\n")
        file.write("Precision score: " + str(precision) + "\n")
        file.write("Recall score: " + str(recall) + "\n")
        file.write("F1 Score: " + str(f1) + "\n")
        file.write("Confusion Matrix: \n")
        file.write(str(conf_matrix))









