setup: requirements.txt
	pip install -r requirements.txt

train: train.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv
	python3 train.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv model.pth

tune: tune.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv
	python3 tune.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv params.config

predict: predict.py data/diabetes_test_analysis.csv data/diabetes_test_info.csv model.pth
	python3 predict.py data/diabetes_test_analysis.csv data/diabetes_test_info.csv model.pth

tune_and_train: tune.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv
	python3 tune.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv params.config -m model.pth

train_config: train.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv params.config
	python3 train.py data/diabetes_train_analysis.csv data/diabetes_train_info.csv model.pth -c params.config

clean:
	rm -rf __pycache__
