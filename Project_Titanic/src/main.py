import click
import yaml
from import_and_clean_data import import_data, clean_data
from model import train_model
from check import check_model


@click.command()
@click.option('--config', '-c', default='../config.yaml', help='Path to the configuration file')
def run_pipeline(config):
    with open(config, 'r') as file:
        config_data = yaml.safe_load(file)

        train_data_path = config_data.get('train_data')
        test_data_path = config_data.get('test_data')

        train_data = import_data.read_data(train_data_path)
        test_data = import_data.read_data(test_data_path)

        X_train, Y_train, clean_data_train = clean_data.clean_data(train_data)
        X_test, Y_test, clean_data_test = clean_data.clean_data(test_data)

        decisiontree_model = train_model.train_decision_tree(X_train, Y_train)
        svm_model = train_model.train_svm(X_train, Y_train)
        random_forest_model = train_model.train_random_forest(X_train, Y_train)

        check_model.check(decisiontree_model, X_test, Y_test,
                          "Decision_Tree_scores")  # last argument is directory to save scores
        check_model.check(svm_model, X_test, Y_test, "Svm_model_scores")  # last argument is directory to save scores
        check_model.check(random_forest_model, X_test, Y_test,
                          "Random_Forest_scores")  # last argument is directory to save scores


if __name__ == "__main__":
    run_pipeline()
