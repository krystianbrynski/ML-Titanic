from import_and_clean_data import import_data , clean_data

Start = 0

while Start !=1 :
       Start = int(input("This is my first Project in python if you want to load data press 1: "))

import_data.read_train_data()
import_data.read_test_data()

train_data = import_data.read_train_data()
test_data = import_data.read_test_data()

X_train, Y_train, clean_data_train = clean_data.clean_train_data(train_data)
X_test, Y_test, clean_data_test = clean_data.clean_test_data(test_data)

