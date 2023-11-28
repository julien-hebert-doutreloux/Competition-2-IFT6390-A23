import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau


class CNN:
    def __init__(self, num_classes, *args, **kwargs):
        # Initialize your custom parameters here
        # Initialize any other variables needed
        self.model = Sequential()
        self.learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
        # Build the CNN model
        self.build_model()

    def build_model(self):
        # Build the CNN model architecture
        self.model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        self.model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        self.model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        self.model.add(Flatten())
        self.model.add(Dense(units = 512 , activation = 'relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units = 24 , activation = 'softmax'))
        self.model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

        
    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        # Convert labels to one-hot encoding
        self.num_classes = np.unique(y_train)
        lb = LabelBinarizer()
        y_train_one_hot = lb.fit_transform(y_train)
        y_val_one_hot = lb.fit_transform(y_val)
        # Train the model
        self.model.fit(X_train, y_train_one_hot, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val_one_hot), callbacks = [self.learning_rate_reduction])

    def predict(self, X):
        # Your prediction logic goes here
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        # Convert labels to one-hot encoding
        lb = LabelBinarizer()
        y_test_one_hot = lb.fit_transform(y_test)

        # Evaluate the model on the test set
        evaluation = self.model.evaluate(X_test, y_test_one_hot)
        return evaluation
        
    def get_params(self, deep=True):
        return {
            'num_classes': self.num_classes,
            # Add any other parameters here
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if hasattr(self, parameter):
                setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter: {parameter}")
