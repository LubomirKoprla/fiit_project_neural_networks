import pandas as pd

def load_events():
    return pd.read_csv("../data/raw/ecommerce-dataset/events.csv")
    
def load_category_tree():
    return pd.read_csv("../data/raw/ecommerce-dataset/category_tree.csv")
    
def load_category_item_properties_part1():
    return pd.read_csv("../data/raw/ecommerce-dataset/item_properties_part1.csv")
    
def load_category_item_properties_part2():
    return pd.read_csv("../data/raw/ecommerce-dataset/item_properties_part2.csv")
    