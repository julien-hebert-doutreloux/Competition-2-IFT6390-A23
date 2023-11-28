import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scripts.RF import *

def experience1():

    # Import data
    data_path = os.path.join('.','data','processed','augmented_flattened_images.pqr')
    train_df = pq.read_table(data_path)    
    train_df = train_df.to_pandas()
    
    train_df = train_df.sample(n=1000)
    train_df = train_df.sample(frac=1)
    train_df.reset_index(drop=True, inplace=True)
    
    y = train_df['label']
    X = train_df.copy().drop(columns=['label'])
    
    # split data
    random_state = 11
    test_size = 0.90
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model = RF(n_estimators=5, max_depth=10, bootstrap_fraction=0.5, features_fraction=0.5)
    
    # fit
    model.fit(X_train.values, y_train.values)
    
    # predict
    y_pred = model.predict(X_test.values)

    print(np.mean(y_test==y_pred))
    # export
experience1()