import pandas as pd


def encodecategorical(data):
    
    '''
    Input
    data: Pandas data frame
    
    Output
    1. categorical value map keys
    2. Updated data using map

    '''
    # get all the columns with object data type
    allcols = data.columns.to_list()
    objectcols = []
    allunq_mapper = []
    for col in allcols:
        if data[col].dtype=="O":
            objectcols.append(col)

    # count the total unique object value and convert them to numeric
    # map data to df
    for col in objectcols:
        unq_mapper = {unq:unq_id+1 for unq_id, unq in enumerate(data[col].unique())}
        data[col] = data[col].map(unq_mapper)
        allunq_mapper.append(unq_mapper)

    return (allunq_mapper, data)
