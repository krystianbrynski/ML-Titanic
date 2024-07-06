# Machine Learning Titanic Project

The aim of this project is to analyse the Titanic dataset and create models that predict which people are more likely to survival. It is a binary classification where the name of the target column is "Survived" indicates whether a person survived 1 or did not survive 0.

I used Jupyter Notebook to analyze the data and select optimal features for the models. I implemented three basic models: Support Vector Machine, Decision Tree, and Random Forest.

Dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data
## Definitions of three basic models:

- **Support Vector Machine (SVM):**
  This model is a powerful classification algorithm that aims to find a hyperplane in multidimensional space that maximizes the margin between different classes.

- **DecisionTreeClassifier:**
  This model is a classification algorithm that creates a tree-like model where each internal node represents a feature or attribute.

- **RandomForestClassifier:**
  This model is an ensemble classification algorithm that utilizes multiple decision trees during training. Each decision tree is trained on different subsets of data and random subsets of features. The final class (for classification) is determined by voting or averaging the predictions of the individual trees, which enhances accuracy and stability.

## Description of files:
- **main.py -**
  Main file where function calls from other files are located.

- **import_data.py -**
  Responsible for loading training and testing data.

- **clean_data.py -**
  Cleans the training and testing data and returns the target column "Survived" and features.

- **train_model.py -**
  Contains 3 functions for training the three models. For Decision Tree and Random Forest models, it saves tree diagrams in the "scores" folder.

- **check_model.py -**
  Checks the results of a given model and saves them to the "scores" folder.

- **data_analysis_titanic.ipynb -** 
  This file contains the data analysis of Titanic dataset.

- **Svm_model_scores.txt** - Results for the SVM model

- **Decision_Tree_scores.txt** - Results for the Decision Tree model

- **Random_Forest_scores.txt** - Results for the Random Forest model

- **requirements.txt** - The libraries used in python

- **config.yaml** - Contains paths to training and testing data, as well as to the scores folder. It also includes parameters for both the Decision Tree and Random Forest models.

## Metrics used to record results:

In the "scores" folder, there are three subfolders dedicated to each specific model, where the following metrics are saved:

- **F1 Score:** Harmonic mean of precision and recall, balancing both metrics.
- **Accuracy:** Overall correctness of the model's predictions.
- **Precision:** Proportion of true positive predictions out of all positive predictions.
- **Recall:** Proportion of true positive predictions out of all actual positive instances.

![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/check/Project_Titanic/photos/Screenshoot.png)

- **True Positives (TP):** The number of correctly predicted positive instances.
- **True Negatives (TN):** The number of correctly predicted negative instances.
- **False Positives (FP):** The number of negative instances predicted as positive.
- **False Negatives (FN):** The number of positive instances predicted as negative.


## Model Ranking Based on Performance

1. Support Vector Machine 
2. Decision Tree Classifier
3. RandomF Forest Classifier

To check detailed results, please navigate to the 'scores' folder.

## Decision tree classifier diagram: 

![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/check/Project_Titanic/scores/Decision_Tree_scores/decision_tree_plot.png)

## Random forest classifier diagrams: 

![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/check/Project_Titanic/scores/Random_Forest_scores/decision_tree_1.png)
![Screenshot](https://github.com/krystianbrynski/KrystianrBry_ML_Titanic/blob/check/Project_Titanic/scores/Random_Forest_scores/decision_tree_2.png)



