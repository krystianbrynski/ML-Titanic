import pandas as pd


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned_data = data
    cleaned_data = cleaned_data.drop(["PassengerId", "Cabin", "Name", "Ticket"], axis=1)

    cleaned_data["0 - 18"] = 0
    cleaned_data["19 - 40"] = 0
    cleaned_data["40+"] = 0

    cleaned_data["0 - 18"] = cleaned_data["0 - 18"].where(~ (data["Age"] <= 18), 1)
    cleaned_data["19 - 40"] = cleaned_data["19 - 40"].where(~ ((data["Age"] > 18) & (data["Age"] <= 40)), 1)
    cleaned_data["40+"] = cleaned_data["40+"].where(~ (data["Age"] > 40), 1)

    cleaned_data["Fare"] = cleaned_data["Fare"].fillna(cleaned_data["Fare"].mean())

    cleaned_data["Sex_binary"] = 0
    cleaned_data["Sex_binary"] = cleaned_data["Sex_binary"].where(data["Sex"] == "male", 1)

    cleaned_data = cleaned_data.drop(["Sex", "Age"], axis=1)

    missing_rows = cleaned_data.query("Embarked.isna()")
    cleaned_data = cleaned_data.drop(missing_rows.index, axis=0)

    x = pd.get_dummies(cleaned_data["Embarked"]).astype("int")
    cleaned_data = cleaned_data.assign(
        C=x["C"],
        Q=x["Q"],
        S=x["S"],
    )

    cleaned_data = cleaned_data.drop("Embarked", axis=1)

    Y = cleaned_data["Survived"]  # Selected target
    X = cleaned_data.drop("Survived", axis=1)  # Selected features, all without target column (Survived)

    return X, Y  # return target, features
