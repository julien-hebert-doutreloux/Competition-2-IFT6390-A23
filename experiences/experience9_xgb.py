import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from scripts.Data import *

sys.path.append('.')

def experience9():

    # Import data
    # data import
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv")
    val_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")

    y_train = train_df['label'].values
    X_train = train_df.copy().drop(columns=['label']).values
    
    y_val = val_df['label'].values
    X_val = val_df.copy().drop(columns=['label']).values

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    num_classes = len(label_encoder.classes_)
    
    # Flattening the images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Normalization
    X_train = X_train / 255.0
    X_val = X_val / 255.0    
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    model = xgb.XGBClassifier(learning_rate = .01, n_estimators = 200, max_depth = 20, objective='multi:softmax', num_class=num_classes)

    # fit
    model.fit(X_train, y_train)
    
    # predict
    y_val_pred = model.predict(X_val)
    y_val_pred[y_val_pred >= 9] +=  1
    
    print(f"Classification Report: \n{classification_report(y_val, y_val_pred)}")

    
    # if False:# prediction on test
    #     test_df = pd.read_csv("./data/raw/test.csv")    
    
    #     # Splitting columns A1 to A784 into a separate DataFrame
    #     df_A = test_df.filter(like='pixel_a')
    #     #test_A = df_A.values.reshape(-1,28,28,1)
    #     y_pred_A = model.predict(df_A.values)
    
    #     # Splitting columns B1 to B784 into a separate DataFrame
    #     df_B = test_df.filter(like='pixel_b')
    #     #test_B = df_B.values.reshape(-1,28,28,1)
    #     y_pred_B = model.predict(df_B.values)
    
    
    #     transformed_labels = [transform_labels(label_A, label_B) for label_A, label_B in zip(y_pred_A, y_pred_B)]
    
    #     IDS = np.arange(len(transformed_labels))
    #     res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})
    
    #     prediction_path = os.path.join('data','prediction','xgboost1.csv')
    #     print(res.head(10))
    #     print(np.unique(res['label'], return_counts=True))
    #     res.to_csv(prediction_path, index=False)
experience9()