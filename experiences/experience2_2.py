import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scripts.CNN import *
from scripts.Data import *
def experience2():

    # Import data
    data_path = os.path.join('.','data','processed','augmented_flattened_images.pqr')
    weights_path = os.path.join('.','data','weights','model_weights.h5')
    train_df = pq.read_table(data_path)
    train_df = train_df.to_pandas()
    #train_df.loc[train_df['label'] >= 9, 'label'] += 1
    train_df = train_df.sample(frac=0.7, random_state=11)
    
    #print(np.unique(train_df['label'], return_counts=True))
    
    y = train_df['label'].values
    X = train_df.copy().drop(columns=['label']).values
    
    # split data
    random_state = 11
    test_size = 0.40
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_size/2, random_state=random_state, stratify=y_val)
    
    X_train = X_train.reshape(-1,28,28,1)
    X_val = X_val.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)
    
    if not os.path.exists(weights_path):
        model = CNN()
        # fit
        model.fit(X_train, y_train, X_val, y_val)
        # export
        model.model.save_weights(weights_path)
    else:
        model = CNN()
        model.model.load_weights(weights_path)
        
    # predict
    y_pred = np.argmax(model.predict(X_test),axis=1)
    y_pred[y_pred >= 9] +=  1
    score = model.evaluate(X_test, y_test)
    
    print(f'score : {score}')
    print(f'accuracy : {np.mean(y_pred == y_test)}')
    
    if True:
        # prediction on test
        test_df = pd.read_csv(os.path.join('data','raw','test.csv'))
        
        # Splitting columns A1 to A784 into a separate DataFrame
        df_A = test_df.filter(like='pixel_a')
        test_A = df_A.values.reshape(-1,28,28,1)
        y_pred_A = np.argmax(model.predict(test_A),axis=1)
        y_pred_A[y_pred_A >= 9] +=  1
        
        # Splitting columns B1 to B784 into a separate DataFrame
        df_B = test_df.filter(like='pixel_b')
        test_B = df_B.values.reshape(-1,28,28,1)
        y_pred_B = np.argmax(model.predict(test_B),axis=1)
        y_pred_B[y_pred_B >= 9] +=  1
        
        transformed_labels = [transform_labels(label_A, label_B) for label_A, label_B in zip(y_pred_A, y_pred_B)]
        
        IDS = np.arange(len(transformed_labels))
        res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})
        
        prediction_path = os.path.join('data','prediction','cnn.csv')
        print(res.head(10))
        print(np.unique(res['label'], return_counts=True))
        res.to_csv(prediction_path, index=False)
experience2()
