import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scripts.RF import *
from scripts.Data import *
def experience1():

    # Import data
    # data import
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")
    old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
    test_df = pd.read_csv("./data/raw/test.csv")
    
    #data_path = os.path.join('.','data','processed','augmented_flattened_images.pqr')
    #train_df = pq.read_table(data_path)    
    #train_df = train_df.to_pandas()
    

    train_df = train_df.sample(frac=1, random_state=11)
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
    
    # prediction on test
    test_df = pd.read_csv(os.path.join('data','raw','test.csv'))
    
    # Splitting columns A1 to A784 into a separate DataFrame
    df_A = test_df.filter(like='pixel_a')
    #test_A = df_A.values.reshape(-1,28,28,1)
    y_pred_A = model.predict(df_A.values)
    
    # Splitting columns B1 to B784 into a separate DataFrame
    df_B = test_df.filter(like='pixel_b')
    #test_B = df_B.values.reshape(-1,28,28,1)
    y_pred_B = model.predict(df_B.values)
    
    
    transformed_labels = [transform_labels(label_A, label_B) for label_A, label_B in zip(y_pred_A, y_pred_B)]
    
    IDS = np.arange(len(transformed_labels))
    res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})
    
    prediction_path = os.path.join('data','prediction','rf.csv')
    print(res.head(10))
    print(np.unique(res['label'], return_counts=True))
    res.to_csv(prediction_path, index=False)
experience1()
