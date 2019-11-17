from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocessing(data, 
                    user_col, 
                    item_col, 
                    verbose=0, 
                    min_items_user=5, 
                    max_items_user=100, 
                    predicted_items=1, 
                    pad_maxlen=100, 
                    pad_padding='post'):
    data_proc = preprocessing_seq(data, user_col, item_col, verbose, min_items_user, max_items_user, predicted_items)
    data_x = pad_sequences(data_proc["seq_items"], maxlen=pad_maxlen, padding=pad_padding)
    data_y = preprocessing_pred(data_proc)
            
    return data_x,data_y

def preprocessing_pred(data_proc, 
                        verbose = 0):
    data_y = []
    processing_counter = 0
    items_dic = {}
    
    unique_items = set()
    for x in data_proc.seq_items:
        for i in x:
            unique_items.add(i)
    for x in data_proc.pred_items:
        unique_items.add(x[0])
    
    for counter,item in enumerate(unique_items):
        items_dic[item] = counter
    
    for user_pred_items in data_proc["pred_items"]:
        processing_counter += 1
        if(verbose != 0 and processing_counter%100 == 0):
            print(processing_counter)
        user_pred = np.zeros(len(items_dic))
        for item in user_pred_items:
            user_pred[items_dic[item]]=1
        data_y.append(user_pred)
        
    return data_y

def preprocessing_seq(data, 
                        user_col, 
                        item_col, 
                        verbose=0, 
                        min_items_user=5, 
                        max_items_user=100, 
                        predicted_items=1):
    processing_counter = 0
    result_data = []
    user_items_inter = data.groupby([user_col])[item_col].nunique()
    user_items_inter = user_items_inter[(user_items_inter > min_items_user) & (user_items_inter < max_items_user)]
    data = data[data[user_col].isin(user_items_inter.index)]
    
    for user in data.visitorid.unique():
        processing_counter += 1
        if(verbose != 0 and processing_counter%1000 == 0):
            print(processing_counter)
        items = data[data[user_col] == user].groupby(item_col).timestamp.max().sort_values()
        seq = items[:-1].index.values
        pred = items[-1:].index.values
        result_data.append([user,seq,pred])
    
    return pd.DataFrame(result_data,columns=["user_id","seq_items","pred_items"])
    
def data_split(data_x,data_y,test_size=0.2):
    return train_test_split( data_x, data_y, test_size=test_size)