# Import libraries
import sys
import yaml
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from datapipeline import cropyieldtraindatapipeline


# train and data pre process
def testdataprocess(fileloc, featsel):
    df = pd.read_csv(filepath_or_buffer=fileloc)
    selected_fetaures = featsel
    df_id = list(df['id'])
    df_selected = df[selected_fetaures]
    df_mapper, df_update = cropyieldtraindatapipeline(df_selected)
    return df_mapper, df_update, df_id


# test dataset
def predict(modeloc, dtest):
    model_xgb = xgb.Booster()
    model_xgb.load_model(modeloc)
    pred = model_xgb.predict(dtest)
    return pred



# system args
test_yamlfileloc = sys.argv[1]

# load the train yaml file
with open(test_yamlfileloc, "r") as yf:
    test_yamlfile = yaml.safe_load(yf)

# test yaml
TESTDATALOC = test_yamlfile['test data']
SELECTEDFEATURES = test_yamlfile['features selected']
MODELLOCTION = test_yamlfile['model location']

# load and process data
df_mapper, df_update, df_ids = testdataprocess(fileloc = TESTDATALOC, 
                                        featsel = SELECTEDFEATURES)

# xgb dmatrix
dtest = xgb.DMatrix(df_update, label=None)

# predict
pred_val = predict(modeloc=MODELLOCTION, dtest=dtest)

# make csv
df_res = pd.DataFrame()
df_res['id'] = df_ids
df_res['yield_tpha'] = pred_val.tolist()

time_format = "%Y-%m-%d %H:%M:%S"
time_now = datetime.now().strftime(format=time_format).replace(':', '-')
file_name_res = f"submission_{time_now}.csv"
df_res.to_csv(f"./results/{file_name_res}", index=False)
