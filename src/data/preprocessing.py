from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def preprocessing(data,
                  user_col,
                  item_col,
                  verbose=0,
                  min_items_user=5,
                  max_items_user=100,
                  min_interaction_item=0,
                  predicted_items=0.2,
                  pad_maxlen=100,
                  pad_padding='post',
                  sparse=True):
    data_proc = preprocessing_seq(data, user_col, item_col, verbose=verbose, min_items_user=min_items_user,
                                  max_items_user=max_items_user, min_interaction_item=min_interaction_item,
                                  predicted_items=predicted_items)
    print("pad_sequences")
    data_x = pad_sequences(data_proc["seq_items"], maxlen=pad_maxlen, padding=pad_padding)
    data_y = preprocessing_pred(data_proc, verbose=verbose, sparse=sparse)

    return data_x, data_y


def preprocessing_pred(data_proc,
                       verbose=0,
                       sparse=True):
    data_y = []
    processing_counter = 0
    items_dic = {}

    unique_items = set()
    for x in data_proc.seq_items:
        for i in x:
            unique_items.add(i)
    for x in data_proc.pred_items:
        for i in x:
            unique_items.add(i)

    for counter, item in enumerate(unique_items):
        items_dic[item] = counter

    if sparse:
        column_idx = []
        row_idx = []
        for i, x in enumerate(data_proc.pred_items):
            processing_counter += 1
            if verbose != 0 and processing_counter % 1000 == 0:
                print("preprocessing_pred: " + str(processing_counter) + " / " + str(len(data_proc["pred_items"])))
            for item in x:
                column_idx.append(items_dic[item])
                row_idx.append(i)
        return csr_matrix((np.ones(len(row_idx)), (row_idx, column_idx)))

    else:
        for user_pred_items in data_proc["pred_items"]:
            processing_counter += 1
            if verbose != 0 and processing_counter % 1000 == 0:
                print("preprocessing_pred: " + str(processing_counter) + " / " + str(len(data_proc["pred_items"])))
            user_pred = np.zeros(len(items_dic))
            for item in user_pred_items:
                user_pred[items_dic[item]] = 1
            data_y.append(user_pred)
        return data_y


def preprocessing_seq(data,
                      user_col,
                      item_col,
                      verbose=0,
                      min_items_user=5,
                      max_items_user=100,
                      min_interaction_item=0,
                      predicted_items=0.2):
    processing_counter = 0
    result_data = []

    item_users_inter = data.groupby([item_col])[user_col].nunique()
    item_users_inter = item_users_inter[item_users_inter >= min_interaction_item]
    data = data[data[item_col].isin(item_users_inter.index)]

    user_items_inter = data.groupby([user_col])[item_col].nunique()
    user_items_inter = user_items_inter[(user_items_inter >= min_items_user) & (user_items_inter <= max_items_user)]
    data = data[data[user_col].isin(user_items_inter.index)]

    for user in data.visitorid.unique():
        processing_counter += 1
        if verbose != 0 and processing_counter % 1000 == 0:
            print("preprocessing_seq: " + str(processing_counter) + " / " + str(len(data.visitorid.unique())))

        items = data[data[user_col] == user].groupby(item_col).timestamp.max().sort_values()
        if predicted_items >= 1:
            seq = items[:predicted_items * (-1)].index.values
            pred = items[predicted_items * (-1):].index.values
        else:
            pred_items_count = int(len(items) * predicted_items)
            if pred_items_count < 1:
                pred_items_count = 1
            seq = items[:pred_items_count * (-1)].index.values
            pred = items[pred_items_count * (-1):].index.values

        result_data.append([user, seq, pred])

    return pd.DataFrame(result_data, columns=["user_id", "seq_items", "pred_items"])


def data_split(data_x, data_y, test_size=0.2):
    return train_test_split(data_x, data_y, test_size=test_size)
