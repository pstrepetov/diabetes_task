import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import json
import joblib
import argparse
import re
import numpy as np
import pickle

cat_features = ['cholesterol', 'gluc'] #categorical features that are not encoded
target = 'diabetes' #target
fill_vals_file = r"./fill_vals.pkl"

def preprocess(analysis_data_path, info_data_path, fill_vals=None):
    """
    Preprocessing function
    Returns dataframe with all features and labels and fill_vals tuple

    :param analysis_data_path: path to file diabetes_..._analysis.csv
    :param info_data_path: path to file diabetes_..._info.csv
    :param fill_vals: values to use for filling nan

    """

    #Loading and Merging dataframes
    df_analysis = pd.read_csv(analysis_data_path)
    df_info = pd.read_csv(info_data_path)
    df = pd.merge(df_info, df_analysis, on='id')

    #representation of pressure in the form of two values
    delimeter = re.compile('(\\\\|/)')
    df[['pressure_1', 'pressure_2']] = df['pressure'].str.split('[\\/\\\\]', expand=True)
    df[['pressure_1', 'pressure_2']] = df[['pressure_1', 'pressure_2']].astype(int)
    df.drop(['id','pressure'], axis=1, inplace=True)

    #removing incorrect data
    cond_1 = (df['pressure_1'] > 350) & (df['pressure_1'] < 1)
    cond_2 = (df['pressure_2'] > 300) & (df['pressure_2'] < 1)
    cond_3 = (df['age'] > 140)

    df.loc[cond_1, 'pressure_1'] = np.nan
    df.loc[cond_2, 'pressure_2'] = np.nan
    df.loc[cond_3, 'age'] = np.nan

    #if we preprocess the test data, then fill in the empty fields with the values obtained on the train
    if not fill_vals:
        pressure_1, pressure_2 = int(df['pressure_1'].mode().iloc[0]), int(df['pressure_2'].mode().iloc[0])
        age = int(df['age'].mean())
        weight = float(df['weight'].median())
        fill_vals = (pressure_1, pressure_2, age, weight)
    else:
        pressure_1, pressure_2, age, weight = fill_vals


    #replacing the empty pressure values with modes
    df['pressure_1'] = df['pressure_1'].fillna(pressure_1)
    df['pressure_2'] = df['pressure_2'].fillna(pressure_2)

    #replacing the empty age values with the average
    df['age'] = df['age'].fillna(age).astype(int)

    #replacing the empty weight values with the median
    df['weight'] = df['weight'].fillna(weight)

    #converting gender to a binary form
    df['gender'] = df['gender'].str[0].replace({'m': 1, 'f': 0})

    return df, fill_vals

def train_catboost_model(train_analysis_file, train_info_file, model_path,
                         config_file=None, fill_vals_path=fill_vals_file):
    """
    Training function
    Trains CatBoostClassifier model and saves it to file "model_path",
        also saves filling values for preprocessing test data 

    :param train_analysis_file: path to file diabetes_train_analysis.csv
    :param train_info_file: path to file diabetes_train_info.csv
    :param model_path: path to file where trained model will be saved
    :param config_file: path to file with hyperparameters for the model
    :param fill_vals_path: path to file where filling values for preprocessing test data will be saved

    """
    
    #Getting data
    df, fill_vals = preprocess(train_analysis_file, train_info_file)
    X = df.drop(target, axis=1)
    y = df[target]

    #Saving data for filling missing values
    with open(fill_vals_path, 'wb') as file:
        pickle.dump(fill_vals, file)
    
    #Loading config if it is given
    if (config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    
        #Creating model with hyperparameters from config
        model = CatBoostClassifier(cat_features = cat_features, **config)
    else:
        model = CatBoostClassifier(cat_features = cat_features)
    
    #Training and saving the model
    model.fit(X, y, early_stopping_rounds=20, verbose=100)
    joblib.dump(model, model_path)

def main():
    """
    Main function
    
    """

    parser = argparse.ArgumentParser(description="Train CatBoost model on diabetes data")
    parser.add_argument('train_analysis_file', type=str,
                        help="Path to the training analysis data file")
    parser.add_argument("train_info_file", type=str,
                        help="Path to the training info data file")
    parser.add_argument("model_path", type=str,
                        help="Path to the file where trained model will be saved")
    parser.add_argument("-v", "--fill_vals_path", type=str,
                        help=f"Path to the file where values for preprocessing will be saved, \
                            default: {fill_vals_file}", default=fill_vals_file)
    parser.add_argument("-c", "--config_file", type=str,
                        help="Path to the config file with CatBoost hyperparameters")

    args = parser.parse_args()

    train_catboost_model(args.train_analysis_file, args.train_info_file, args.model_path,
                         args.config_file, args.fill_vals_path,)

if __name__ == "__main__":
    main()

