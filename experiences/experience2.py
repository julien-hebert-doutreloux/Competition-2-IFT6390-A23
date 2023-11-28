import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scripts.CNN import *

def experience2():

    # Import data
    data_path = os.path.join('.','data','processed','augmented_flattened_images.pqr')
    weights_path = os.path.join('.','data','weights','model_weights.h5')
    train_df = pq.read_table(data_path)
    train_df = train_df.to_pandas()
    train_df = train_df.sample(frac=0.10, random_state=11)

    y = train_df['label'].values
    X = train_df.copy().drop(columns=['label']).values
    
    # split data
    random_state = 11
    test_size = 0.20
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_train = X_train.reshape(-1,28,28,1)
    X_val = X_val.reshape(-1,28,28,1)
    
    if not os.path.exists(weights_path):
        model = CNN()
        # fit
        model.fit(X_train, y_train, X_val, y_val)
        # export
        model.model.save_weights(weights_path)
    else:
        model = CNN()
        model.model.load_weights('model_weights.h5')
    # predict
    y_pred = model.predict(X_val)
    score = model.evaluate(X_val, y_val)
    
    print(f'score : {score}')
    print(np.argmax(y_pred,axis=1))
    print(f'accuracy : {np.mean(np.argmax(y_pred,axis=1) == y_val)}')
    
    
experience2()
