import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
import joblib
import argparse
import pickle
from train import preprocess, target, fill_vals_file

def predict(test_analysis_file, test_info_file, model_path, fill_vals_path=fill_vals_file):
    """
    Function making predict
    Prints f1_score 

    :param test_analysis_file: path to file diabetes_test_analysis.csv
    :param test_info_file: path to file diabetes_test_info.csv
    :param model_path: path to file where trained model located
    :param fill_vals_path: path to file with filling values for preprocessing data

    """
    
    #Getting data
    with open(fill_vals_path, 'rb') as file:
        fill_vals = pickle.load(file)
    df, _ = preprocess(test_analysis_file, test_info_file, fill_vals)
    X = df.drop(target, axis=1)
    y = df[target]
    
    #Loading model
    model = joblib.load(model_path)
    
    #Predicting values, calculating f1_score and printing it
    predictions = model.predict(X)
    f1 = f1_score(y, predictions)
    
    print(f"F1 Score on test data: {f1}")

def main():
    """
    Main function
    
    """

    parser = argparse.ArgumentParser(description="Predict target with CatBoost model")
    parser.add_argument("test_analysis_file", type=str, help="Path to the test analysis data file")
    parser.add_argument("test_info_file", type=str, help="Path to the test info data file")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("-v", "--fill_vals_path", type=str, required=False,
                        help=f"Path to the file where values for preprocessing are saved, \
                            default: {fill_vals_file}", default=fill_vals_file)

    args = parser.parse_args()

    predict(args.test_analysis_file, args.test_info_file, args.model_path, args.fill_vals_path)

if __name__ == "__main__":
    main()

