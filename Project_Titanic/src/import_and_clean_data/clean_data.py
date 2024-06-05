import pandas as pd

def clean_data(data):
    data = data.drop("PassengerId", axis=1)
    data = data.drop("Cabin", axis=1)
    data = data.drop("Name", axis=1)
    data = data.drop("Ticket", axis=1)

    data["Age"] = data["Age"].fillna(data["Age"].mean())
    data["Age"] = data["Age"].astype("int")

    data["Sex_binary"] = 0
    data["Sex_binary"] = data["Sex_binary"].where(data["Sex"] == "male", 1)
    data = data.drop("Sex", axis=1)

    x = pd.get_dummies(data["Embarked"]).astype("int")
    data=data.assign(
    C = x["C"],
    Q = x["Q"],
    S = x["S"],
    )

    data = data.drop("Embarked", axis=1)

    #Y = data["Survived"]
    #X = data.iloc[:,1:]

    return (data)

def replace_NAN_ticket(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
    return data