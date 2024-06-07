from import_and_clean_data import import_data , clean_data
from model import train_model
from check import check_model

def model_check(best_model,X_test,Y_test):
       check_model.check(best_model, X_test, Y_test)

def model_train(X_train,Y_train):
       best_model = train_model.train(X_train, Y_train)

       return best_model

def import_and_clean():
       train_data = import_data.read_train_data()
       test_data = import_data.read_test_data()

       X_train, Y_train, clean_data_train = clean_data.clean_train_data(train_data)
       X_test, Y_test, clean_data_test = clean_data.clean_test_data(test_data)

       return X_train, Y_train,X_test, Y_test, clean_data_train

def main():
       X_train, Y_train,X_test, Y_test, clean_data_train = import_and_clean()
       best_model = model_train(X_train,Y_train)
       model_check(best_model,X_test,Y_test)


main()