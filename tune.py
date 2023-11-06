import numpy as np
import optuna
from optuna.integration import CatBoostPruningCallback
import pandas as pd
import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import argparse
import json
import train
from train import preprocess, target, cat_features

num_trials = 15

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Function that performs a single optuna iteration
    Returns f1 score on validation data

    :param trial: optuna trial
    :param X_train: dataframe with train features 
    :param y_train: series object with train labels
    :param X_val: dataframe with validation features 
    :param y_val: series object with validation labels

    """
    
    #creating parameter set
    param = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10),
        "eval_metric": "F1",
    }

    gbm = cb.CatBoostClassifier(cat_features = cat_features, **param)

    #creating pruning callback and fitting model
    pruning_callback = CatBoostPruningCallback(trial, "F1")

    gbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=0,
        callbacks=[pruning_callback],
    )

    pruning_callback.check_pruned()

    #Calculating f1_score
    preds = gbm.predict(X_val)
    f1 = f1_score(y_val, preds)

    return f1

def tune_catboost_hyperparameters(train_analysis_file, train_info_file,
                                  output_config_file, n_trials=num_trials,
                                  model_path=None, fill_vals_path=None):
    """
    Function that tunes hyperparameters for CatBoostClassifier
    Saves best hyperparameters in config file and can launch training on these hyperparameters

    :param train_analysis_file: path to file diabetes_train_analysis.csv
    :param train_info_file: path to file diabetes_train_info.csv
    :param output_config_file: path to file where best hyperparameters will be saved
    :param n_trials: number of optuna trials to run
    :param model_path: path to file where trained model will be saved
    :param fill_vals_path: path to file where filling values for preprocessing test data will be saved
                           if model_path is given

    """
    
    df, _ = preprocess(train_analysis_file, train_info_file)
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    #Optuna study
    study = optuna.create_study(direction='maximize')
    objective_partial = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    study.optimize(objective_partial, n_trials=n_trials)

    #Getting best params from study
    best_params = study.best_params

    #Saving best params
    with open(output_config_file, 'w') as f:
        json.dump(best_params, f, indent=4)

    #Launching training if model_path is given
    if (model_path):
        train.train_catboost_model(train_analysis_file, train_info_file, model_path,
                                   config_file = output_config_file, fill_vals_path = fill_vals_path)

    

def main():
    """
    Main function
    
    """

    parser = argparse.ArgumentParser(description="Tune CatBoost model on diabetes data")
    parser.add_argument("train_analysis_file", type=str,
                        help="Path to the training analysis data file")
    parser.add_argument("train_info_file", type=str,
                        help="Path to the training info data file")
    parser.add_argument("config_file", type=str,
                        help="Path to the output config file with best CatBoost hyperparameters")
    parser.add_argument("-n", "--n_trials", type=int,
                        help=f"int n_trials parameter for optuna, default: {num_trials}", default=num_trials)
    parser.add_argument("-m", "--model_path", type=str,
                        help="Path to the file where model trained on best hyperparameters \
                              will be saved")
    parser.add_argument("-v", "--fill_vals_path", type=str,
                        help=f"Path to the file where values for preprocessing will be saved \
                            if --model_path is not empty, default: {train.fill_vals_file}",
                            default=train.fill_vals_file)

    args = parser.parse_args()

    tune_catboost_hyperparameters(args.train_analysis_file, args.train_info_file,
                                  args.config_file, args.n_trials, args.model_path, args.fill_vals_path)

if __name__ == "__main__":
    main()
