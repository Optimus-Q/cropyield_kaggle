# Import libraries
import sys
import logging
import yaml
import optuna
import pandas as pd
import xgboost as xgb
from datetime import datetime
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from datapipeline import cropyieldtraindatapipeline


# train and data pre process
def traindataprocess(fileloc, featsel):
    df = pd.read_csv(filepath_or_buffer=fileloc)
    selected_fetaures = featsel
    df_selected = df[selected_fetaures]
    df_mapper, df_update = cropyieldtraindatapipeline(df_selected)
    return df_mapper, df_update


# hyper parameter optuna objective
def objective(trial):

    # split data and target into train and validation
    xtrain, xval, ytrain, yval = train_test_split(DATA, TARGET, test_size = OPTUNATRAINDATASIZE, 
                                                  random_state = OPTUNADATASPLITRANDOM)

    # Dmatrix
    dtrain = xgb.DMatrix(xtrain, ytrain)
    dval = xgb.DMatrix(xval, yval)
    
    # all optimisation parameters
    param = {
            "objective": MODELOBJECTIVE,
            "tree_method": MODELTREEMETHOD,
            "learning_rate": trial.suggest_categorical("learning_rate", [0.008,0.01,0.012,0.014,0.016,0.018]),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "alpha": trial.suggest_float("alpha", 0, 10),
            "lambda": trial.suggest_float("lambda", 0, 10), 
            "seed": MODELSEED
            }

    model = xgb.train(params=param, dtrain=dtrain, 
                      num_boost_round=MODELESTIMATORS, evals=[(dval, "val")], 
                      early_stopping_rounds=MODELEARLYSTOPPING, verbose_eval=False)
    preds = model.predict(dval)
    rmse = root_mean_squared_error(yval, preds)
    return rmse


def train(optunastudy, data, target):
     optuna_params = optunastudy.best_trial.params
     optuna_params.update({"objective": MODELOBJECTIVE, "seed": MODELSEED, 
                           "tree_method": MODELTREEMETHOD})
     xtrain, xval, ytrain, yval = train_test_split(data, target, test_size = OPTUNATRAINDATASIZE, 
                                                  random_state = OPTUNADATASPLITRANDOM)
     dtrain = xgb.DMatrix(xtrain, ytrain)
     dval = xgb.DMatrix(xval, yval)
     train_xgb = xgb.train(params=optuna_params,
                            dtrain=dtrain,
                            num_boost_round=MODELESTIMATORS,
                            evals=[(dval, "val")],
                            early_stopping_rounds=MODELEARLYSTOPPING,
                            verbose_eval=False)
     return train_xgb, optuna_params


def test(optunaxgb, data, target):
     deval = xgb.DMatrix(data, label=None)
     predict = optunaxgb.predict(deval)
     rmse = root_mean_squared_error(target, predict)
     return rmse


# Configure logger
logging.basicConfig(
    filename="./train/training logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# system args
train_yamlfileloc = sys.argv[1]

# load the train yaml file
with open(train_yamlfileloc, "r") as yf:
    train_yamlfile = yaml.safe_load(yf)

# main training data parameters
TRAINDATALOC = train_yamlfile['train data']
SELECTEDFEATURES = train_yamlfile['features selected']
TRAINDATASIZE = train_yamlfile['traindatasplit'][0]['traindatasize']
DATASPLITRANDOMSEED = train_yamlfile['traindatasplit'][1]['splitrandomseed']

# optuna data parameters
OPTUNATRAINDATASIZE = train_yamlfile['optunadatasplit'][0]['traindatasize']
OPTUNADATASPLITRANDOM = train_yamlfile['optunadatasplit'][1]['splitrandomseed']

# base model parameters
MODELOBJECTIVE = train_yamlfile['basemodelparameters'][0]['objective']
MODELTREEMETHOD = train_yamlfile['basemodelparameters'][1]['tree_method']
MODELESTIMATORS = train_yamlfile['basemodelparameters'][2]['n_estimators']
MODELEARLYSTOPPING = train_yamlfile['basemodelparameters'][3]['early_stopping']
MODELSEED = train_yamlfile['basemodelparameters'][4]['seed']

# optuna config
OPTUNASAMPLERSEED = train_yamlfile['optunaconfig'][0]['samplerseed']
OPTUNADIRECTION = train_yamlfile['optunaconfig'][1]['optunadirection']
OPTUNANTRIAL = train_yamlfile['optunaconfig'][2]['n_trial']

# load and process data
df_mapper, df_update = traindataprocess(fileloc = TRAINDATALOC, 
                                        featsel = SELECTEDFEATURES)
x = df_update.iloc[:, :-1]
y = df_update.iloc[:, -1]
xtrainval, xeval, ytrainval, yeval = train_test_split(x, y, train_size=TRAINDATASIZE, 
                                                          random_state=DATASPLITRANDOMSEED)
DATA = xtrainval
TARGET = ytrainval

# optuna
study = optuna.create_study(direction=OPTUNADIRECTION, sampler=optuna.samplers.TPESampler(seed=OPTUNASAMPLERSEED))
study.optimize(objective, n_trials=OPTUNANTRIAL)

# run train
train_xgb_model, optuna_optimised_params = train(optunastudy=study, data=xtrainval, target=ytrainval)

# run test
test_rmse_optuna = test(optunaxgb=train_xgb_model, data=xeval, target=yeval)

# optuna best parameter and model save details
time_format = "%Y-%m-%d %H:%M:%S"
optuna_rmse_value = study.best_trial.values[0]
optuna_start_time = study.best_trial.datetime_start.strftime(format=time_format)
optuna_end_time = study.best_trial.datetime_complete.strftime(format=time_format)
optuna_start_time_dt = datetime.strptime(optuna_start_time, time_format)
optuna_end_time_dt = datetime.strptime(optuna_end_time, time_format)
optuna_optim_diff_seconds = (optuna_end_time_dt - optuna_start_time_dt).total_seconds()
trial_model_file_name = f"xgb_base_model_{optuna_end_time.replace(':', '-')}"

# save train model
train_xgb_model.save_model(f"./train/model weights/{trial_model_file_name}.json")

# save trial model - train parameters
with open(f"./train/model parameters/{trial_model_file_name}.txt", "w") as f:
    f.write(str(optuna_optimised_params))

# Log all details
logging.info(
    "Optuna Training Summary | Start: %s | End: %s | Duration: %.2f seconds | OPTUNA RMSE: %.4f | TEST RMSE: %.4f | Model: %s",
    optuna_start_time,
    optuna_end_time,
    optuna_optim_diff_seconds,
    optuna_rmse_value,
    test_rmse_optuna,
    trial_model_file_name)
