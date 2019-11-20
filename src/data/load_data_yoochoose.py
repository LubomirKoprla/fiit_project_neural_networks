import pandas as pd

def load_clicks():
    return pd.read_csv("../data/raw/yoochoose-csv/yoochoose-clicks.csv",names=["visitorid","timestamp","itemid","category"])
    
def load_purchases():
    return pd.read_csv("../data/raw/yoochoose-csv/yoochoose-buys.csv",names=["visitorid","timestamp","itemid","price","quantity"])

def load_test():
    return pd.read_csv("../data/raw/yoochoose-csv/yoochoose-test.csv",names=["visitorid","timestamp","itemid","category"])