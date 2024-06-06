import pandas as pd

def clean_train_data(data):
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)

    data["0 - 18"] = 0
    data["19 - 40"] = 0
    data["40+"] = 0

    data["0 - 18"] = data["0 - 18"].where(~ (data["Age"] <= 18), 1)
    data["19 - 40"] = data["19 - 40"].where(~ ((data["Age"] > 18) & (data["Age"] <= 40)), 1)
    data["40+"] = data["40+"].where(~ (data["Age"] > 40), 1)

    data["Sex_binary"] = 0
    data["Sex_binary"] = data["Sex_binary"].where(data["Sex"] == "male", 1)

    data = data.drop("Sex", axis=1)
    data = data.drop("Age", axis=1)

    missing_rows = data.query("Embarked.isna()")
    data = data.drop(missing_rows.index, axis=0)

    x = pd.get_dummies(data["Embarked"]).astype("int")
    data=data.assign(
    C = x["C"],
    Q = x["Q"],
    S = x["S"],
    )

    data = data.drop("Embarked", axis=1)

    Y = data["Survived"]
    X= data.iloc[:,1:]

    return (X,Y,data)

def clean_test_data(data):
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)


    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

    data["0 - 18"] = 0
    data["19 - 40"] = 0
    data["40+"] = 0

    data["0 - 18"] = data["0 - 18"].where(~ (data["Age"] <= 18), 1)
    data["19 - 40"] = data["19 - 40"].where(~ ((data["Age"] > 18) & (data["Age"] <= 40)), 1)
    data["40+"] = data["40+"].where(~ (data["Age"] > 40), 1)



    data["Sex_binary"] = 0
    data["Sex_binary"] = data["Sex_binary"].where(data["Sex"] == "male", 1)

    data = data.drop("Sex", axis=1)
    data = data.drop("Age", axis=1)

    x = pd.get_dummies(data["Embarked"]).astype("int")
    data=data.assign(
    C = x["C"],
    Q = x["Q"],
    S = x["S"],
    )

    data = data.drop("Embarked", axis=1)

    Y = data["Survived"]
    X = data.iloc[:,1:]

    return (X,Y,data)
