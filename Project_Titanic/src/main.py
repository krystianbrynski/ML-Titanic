from import_and_clean_data import import_data , clean_data

Start = 0

while Start !=1 :
       Start = int(input("This is my first Project in python if you want to load data press 1: "))

train_data = import_data.read_train_data()
test_data = import_data.read_test_data()

clean_data_train = clean_data.clean_data(train_data)

almost_clean_data_test = clean_data.clean_data(test_data)
clean_data_test = clean_data.replace_NAN_ticket(almost_clean_data_test)

X_train = clean_data_train["Survived"]
Y_train = clean_data_train.iloc[:,1:]

X_test = clean_data_test["Survived"]
Y_test = clean_data_test.iloc[:,1:]

