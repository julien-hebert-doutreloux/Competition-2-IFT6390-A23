from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
import pyarrow as pa
import pyarrow.parquet as pq
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('.')
from scripts.RF import *
from scripts.Data import *



random_state = 2


# data import
old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
train_df = pd.read_csv("./data/raw/sign_mnist_train.csv").sample(frac=1)
test_df = pd.read_csv("./data/raw/test.csv")

# Data manipulation
y_train = train_df['label']
y_test = old_test_df['label']

# Normalization and negative
X_train = train_df.drop(columns=['label']).values/255
X_test = old_test_df.drop(columns=['label']).values/255


def reshape_sum_and_fft(row):
    # Reshape the row to 28x28 matrix
    reshaped_row = row.reshape(28, 28)
    
    # Sum along the rows (axis=0)
    row_sum = np.sum(reshaped_row, axis=0)

    # Sum along the rows (axis=0)
    col_sum = np.sum(reshaped_row, axis=1)
    
    # Apply FFT on the summed row
    row_fft = fft(row_sum)
    col_fft = fft(col_sum)
    res_fft = row_fft*col_fft
    
    return  res_fft.real+res_fft.imag


# Apply the function to each row of X_train
X_train_fft = np.apply_along_axis(reshape_sum_and_fft, 1, X_train)
X_test_fft = np.apply_along_axis(reshape_sum_and_fft, 1, X_test)


# Split the data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X_train_fft, y_train, test_size=0.40, random_state=random_state)

best_params = {'n_estimators': 178, 'max_depth': 90}
model =  RF(**best_params)

# Train the classifier
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_fft)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))
print(f"Accuracy : {accuracy}")

formatted_number = "{:.4f}".format(accuracy).replace('.','_')
name_tag = f'{random_state}_{formatted_number}'

# Plotting the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

figure_path = os.path.join('.','figures', f'RF_scratch_{name_tag}.pdf')
plt.savefig(figure_path)


# Kaggle prediction
df_A = test_df.filter(like='pixel_a')
test_A = df_A.values.reshape(-1,28,28,1)

df_B = test_df.filter(like='pixel_b')
test_B = df_B.values.reshape(-1,28,28,1)

y_pred_A = model.predict(test_A)
y_pred_A[y_pred_A >= 9] +=  1
y_pred_B = model.predict(test_B)
y_pred_B[y_pred_B >= 9] +=  1

# ascii sum
y_pred = transform_labels(y_pred_A,y_pred_B)
    
# export prediction
file_path = os.path.join('.','data','prediction')
file_name = f'rf_scratch_{name_tag}'
export_prediction(y_pred, file_path, file_name)



