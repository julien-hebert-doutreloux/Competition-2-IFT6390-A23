import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('.')

def preprocess_images(images):
    resized_images = []
    for img in images:
        # Reshape to 2D grid (assuming original shape is square)
        img_2d = img.reshape(int(img.shape[0]**0.5), int(img.shape[0]**0.5))
        # Resize using scikit-image
        resized_img = resize(img_2d, (img_2d.shape[0] // 4, img_2d.shape[1] // 4), anti_aliasing=True)
        # Flatten the resized image
        resized_images.append(resized_img.flatten())
        
    return np.array(resized_images)
    
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
    
    # Reshape and resize the images and flatten them through the preprocess function (cutting resolution by a factor of .25)
    X_train = preprocess_images(X_train)
    X_val = preprocess_images(X_val)
    
    # Flattening the images
    #X_train = X_train.reshape(X_train.shape[0], -1)
    #X_val = X_val.reshape(X_val.shape[0], -1)
    
    # Normalization
    X_train = X_train / 255.0
    X_val = X_val / 255.0    
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [8, 10, 12],    
    }

    
    # Create an XGBoost classifier for multiclass classification
    xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=num_classes, eval_metric='mlogloss')

    # Perform RandomizedSearchCV with k-fold Cross Validation
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(xgb_classifier, param_distributions=param_dist, scoring='accuracy', n_jobs=-1, cv=k_fold, random_state=42)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
    model = XGBClassifier(random_state = 42, **best_params)
    
    # fit
    model.fit(X_train, y_train)
    
    # predict
    y_val_pred = model.predict(X_val)
    print(y_val_pred)
    #y_val_pred[y_val_pred >= 9] +=  1
    
    print(f"Classification Report: \n{classification_report(y_val, y_val_pred)}")
    
    results = random_search.cv_results_
    print(f"Best Hyperparameters: \n{random_search.best_params_}")
    #print(f"results: \n{results}")
    
    # Extract hyperparameter sets and corresponding metrics
    hyperparameter_sets = [str(params) for params in results['params']]
    accuracy_scores = results['mean_test_score']
    
    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'Hyperparameter Set': hyperparameter_sets,
        'Accuracy': accuracy_scores
    })
    
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Accuracy', y='Hyperparameter Set', data=df, color='skyblue')
    plt.xlim(0.90, 1.0)
    plt.yticks(range(len(hyperparameter_sets)), hyperparameter_sets, rotation=None)
    plt.title(f'Accuracy vs HPs')
    plt.tight_layout()
    
    # Save the figure as a PNG file
    output_filepath = os.path.join('.','data','hyperparameters',f'xgb_hp_Accuracy.png')
    plt.savefig(output_filepath)
    plt.close()  # Close the current figure to free up memory
    
    # Plot and save each metric as a PNG file
    # for metric in ['Precision', 'F1 Score', 'Recall', 'Accuracy']:
    #     plt.figure(figsize=(8, 6))
    #     sns.barplot(x=metric, y='Hyperparameter Set', data=df, color='skyblue')
    #     plt.xlim(0.7, .85)
    #     plt.yticks(range(len(param_vals)), param_vals, rotation=None)
    #     plt.title(f'{metric} vs HPs')
    #     plt.tight_layout()
    
    #     # Save the figure as a PNG file
    #     output_filepath = os.path.join('.','data','hyperparameters',f'xgb_hp_{metric}.png')
    #     plt.savefig(output_filepath)
    #     plt.close()  # Close the current figure to free up memory

    
    if True:# prediction on test
        
        test_df = pd.read_csv("./data/raw/test.csv")    
        test_df = test_df.drop(columns=['id'])
        # Splitting columns A1 to A784 into a separate DataFrame
        df_A = test_df.filter(like='pixel_a')
        test_A = preprocess_images(df_A.values)
        test_A = test_A/255.0
        test_A = scaler.transform(test_A)

        y_pred_A = model.predict(test_A)
        y_pred_A[y_pred_A >= 9] +=  1
    
        # Splitting columns B1 to B784 into a separate DataFrame
        df_B = test_df.filter(like='pixel_b')
        test_B = preprocess_images(df_B.values)
        test_B = test_B/255.0
        test_B = scaler.transform(test_B)

        y_pred_B = model.predict(test_B)
        y_pred_B[y_pred_B >= 9] +=  1    
        
    
        # ascii sum
        y_pred = transform_labels(y_pred_A,y_pred_B)

        # export prediction
        file_path = os.path.join('.','data','prediction')
        file_name = f'xgboost_finale'
        export_prediction(y_pred, file_path, file_name)
        
        
experience9()
