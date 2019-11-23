import pandas as pd
import numpy as np
import scipy.sparse


def load_clicks():
    return pd.read_csv("../../data/raw/yoochoose-csv/yoochoose-clicks.csv",
                       names=["visitorid", "timestamp", "itemid", "category"])

def load_purchases():
    return pd.read_csv("../../data/raw/yoochoose-csv/yoochoose-buys.csv",
                       names=["visitorid", "timestamp", "itemid", "price", "quantity"])

def load_test():
    return pd.read_csv("../../data/raw/yoochoose-csv/yoochoose-test.csv",
                       names=["visitorid", "timestamp", "itemid", "category"])

def save_processed(data_x, data_y):
    np.savetxt("../../data/processed/yoochoose/dataset_v1/data_x.csv", np.asarray(data_x), delimiter=",")
    np.savetxt("../../data/processed/yoochoose/dataset_v1/data_y.csv", np.asarray(data_y), delimiter=",")

def load_processed():
    data_x = np.genfromtxt("../../data/processed/yoochoose/dataset_v1/data_x.csv", delimiter=",")
    data_y = np.genfromtxt("../../data/processed/yoochoose/dataset_v1/data_y.csv", delimiter=",")
    return data_x, data_y

def save_processed_sparse(data_x, data_y):
    #np.savetxt("../../data/processed/yoochoose/dataset_v1_sparse/data_x.csv", np.asarray(data_x), delimiter=",")
    data_x = pd.read_csv("../../data/processed/yoochoose/data_x.csv", header=None).values
    scipy.sparse.save_npz('../../data/processed/yoochoose/dataset_v1_sparse/data_y.npz', data_y)

def load_processed_sparse():
    data_x = np.genfromtxt("../../data/processed/yoochoose/dataset_v1_sparse/data_x.csv", delimiter=",")
    data_y = scipy.sparse.load_npz("../../data/processed/yoochoose/dataset_v1_sparse/data_y.npz")
    return data_x, data_y

def transform_sparse_to_array(data_y):
    return data_y.toarray()
