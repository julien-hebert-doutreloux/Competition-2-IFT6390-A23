import numpy as np
import pandas as pd
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.callbacks import EarlyStopping

sys.path.append('.')
from scripts.Data import *


import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process random_state_int and weight_file_path.')
    parser.add_argument('--random_state_int', type=int, help='Random state integer')
    parser.add_argument('--weight_file_path', type=str, help='Path to the weight file')

    args = parser.parse_args()

    random_state = int(args.random_state_int) if args.random_state_int else 42
    weights_path = args.weight_file_path

    # data import
    old_test_df = pd.read_csv("./data/raw/old_sign_mnist_test.csv")
    test_df = pd.read_csv("./data/raw/test.csv")
    train_df = pd.read_csv("./data/raw/sign_mnist_train.csv").sample(frac=1)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=random_state)

    # Data manipulation
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = old_test_df['label']

    # label
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_val = label_binarizer.fit_transform(y_val)

    # Normalization
    X_train = train_df.drop(columns=['label']).values
    X_val = val_df.drop(columns=['label']).values/255
    X_test = old_test_df.drop(columns=['label']).values/255

    # Reshape
    X_train = X_train.reshape(-1,28,28,1)
    X_val = X_val.reshape(-1,28,28,1)
    X_test = X_test.reshape(-1,28,28,1)

    # Data augmentation
    # Modele 1
    datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images  
        brightness_range=[0.15,.65])
    datagen.fit(X_train)


    # Modele de base
    model = Sequential()
    model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 512 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units = 24 , activation = 'softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 5, verbose=1, factor=0.5, min_lr=0.00001)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


    model = tf.keras.models.clone_model(model)
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


    directory = './data/weights/'
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter only files with specific extensions (e.g., '.h5', '.weights', etc.)
    model_weights_path = [os.path.join(directory, file) for file in files]
    if False:
        model_weights_path =     [
        './data/weights/model_weights_14_17_0_9922.h5',
        './data/weights/model_weights_13_11_0_9890.h5',
        './data/weights/model_weights_12_14_0_9907.h5',
        './data/weights/model_weights_11_20_0_9921.h5',
        './data/weights/model_weights_10_21_0_9974.h5',
        './data/weights/model_weights_9_14_0_9944.h5',
        './data/weights/model_weights_8_10_0_9827.h5',
        './data/weights/model_weights_7_18_0_9960.h5',
        './data/weights/model_weights_6_14_0_9911.h5',
        './data/weights/model_weights_5_16_0_9994.h5',
        './data/weights/model_weights_4_17_0_9960.h5',
        './data/weights/model_weights_2_22_0_9971.h5',
        './data/weights/model_weights_1_19_0_9927.h5',
        ]

    model_dict = {path: tf.keras.models.clone_model(model) for path in model_weights_path}
    for p in model_weights_path:
        if not '.gitkeep' in p: 
            model_dict[p].compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
            model_dict[p].load_weights(p)
        
        
    all_predictions_A = []
    all_predictions_B = []
    
    df_A = test_df.filter(like='pixel_a')
    test_A = df_A.values.reshape(-1,28,28,1)
    df_B = test_df.filter(like='pixel_b')
    test_B = df_B.values.reshape(-1,28,28,1)
    from collections import Counter
    for i, md in enumerate(model_dict.values()):
    
        y_pred_A = np.argmax(md.predict(test_A), axis=1)
        y_pred_A[y_pred_A >= 9] +=  1
        
        y_pred_B = np.argmax(md.predict(test_B),axis=1)
        y_pred_B[y_pred_B >= 9] +=  1
        
        all_predictions_A.append(y_pred_A)
        all_predictions_B.append(y_pred_B)
        
    # Perform majority voting for component A
    final_predictions_A = []

    # Compute majority vote for component A
    num_samples_A = len(all_predictions_A[0])

    for sample_index in range(num_samples_A):
        component_votes_A = [pred[sample_index] for pred in all_predictions_A]
        majority_vote_A = Counter(component_votes_A).most_common(1)[0][0]
        final_predictions_A.append(majority_vote_A)

    # Perform majority voting for component B
    final_predictions_B = []

    # Compute majority vote for component B
    num_samples_B = len(all_predictions_B[0])

    for sample_index in range(num_samples_B):
        component_votes_B = [pred[sample_index] for pred in all_predictions_B]
        majority_vote_B = Counter(component_votes_B).most_common(1)[0][0]
        final_predictions_B.append(majority_vote_B)

    # Print the final predictions for each component
    print("Final predictions for Component A:", final_predictions_A)
    print("Final predictions for Component B:", final_predictions_B)
    

        
       

    def label_to_uppercase(index):
        return chr(index + 65)  # 'A' is ASCII 65

    # Normalize ASCII sum
    def normalize_ascii_sum(ascii_sum):
        while ascii_sum > 122:  # 'z' is ASCII 122
            ascii_sum -= 65  # 122 ('z') - 65 ('A') + 1
        return ascii_sum

    # Convert predictions to uppercase letters and then to ASCII
    ascii_predictions_a = [ord(label_to_uppercase(p)) for p in final_predictions_A]
    ascii_predictions_b = [ord(label_to_uppercase(p)) for p in final_predictions_B]

    # Sum and normalize ASCII values
    ascii_sums = [normalize_ascii_sum(a + b) for a, b in zip(ascii_predictions_a, ascii_predictions_b)]

    # Convert sums to characters
    transformed_labels = [chr(ascii_sum) for ascii_sum in ascii_sums]

    #transformed_labels = [transform_labels(label_A, label_B) for label_A, label_B in zip(y_pred_A,y_pred_B)]
    IDS = np.arange(len(transformed_labels))
    res = pd.DataFrame(data={"id":IDS, 'label':transformed_labels})

    prediction_path = os.path.join('data','prediction','cnn_1.csv')
    print(np.unique(res['label'], return_counts=True))
    res.to_csv(prediction_path, index=False)



