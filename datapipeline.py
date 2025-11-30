from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from dataprocess import encodecategorical
import pandas as pd


def harvestdatetime(data):

    '''data: Input dataframe with harvest date field, which is in object form, 
            we will transform it to pandas datetime'''
    
    data["harvest_date"] = pd.to_datetime(data["harvest_date"], format = "%Y-%m-%d")
    return data

def featureengineering(data):
    '''data: input dataframe'''
    data['harvest_month'] = data["harvest_date"].dt.month
    return data

def cropyieldatapipeline(data):
    
    '''data: input dataframe'''
    harvestfunction = FunctionTransformer(harvestdatetime)
    featurefunction = FunctionTransformer(featureengineering)
    encodefunction = FunctionTransformer(encodecategorical)
    cropyieldpipeline = Pipeline([('Harvest Datetime', harvestfunction),
                                  ('Feature Engineering', featurefunction),
                                  ('Encoding Categorical', encodefunction)])
    processeddata = cropyieldpipeline.fit_transform(data)
    return processeddata

def cropyieldtraindatapipeline(data):

    '''data: input data'''

    encodefunction = FunctionTransformer(encodecategorical)
    cropyieldpipeline = Pipeline([('Encoding Categorical', encodefunction)])
    processeddata = cropyieldpipeline.fit_transform(data)
    return processeddata