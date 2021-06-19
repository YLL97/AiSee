"""
1D input training example steps

Dataset structure:

Number of training datasets*: 2000 (Age : Side effect)      //including validation sets
Number of testing datasets: 200 (10% of training set)

Input data = 1D array with ages from 13-100
Class/Label = has side effect (1) and no side effect (0)

95% of 13-64 aged patients has no side effect (labeled as 0)
95% of 65-100 aged patients has side effect (labeled as 1)

Modified from: https://www.youtube.com/watch?v=qFJeN9V1ZsI
"""

import pathlib
import os
import numpy as np
from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers             # Activation, Dense
from tensorflow.keras import optimizers         # Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from confusion_matrix_plot import plot_confusion_matrix


train_x = []  # list of input datas for training the model
train_y = []  # list of targets/classes/labels associate with each training input datas
test_x = []  # list of input datas for inference using trained model
test_y = []  # list of known  targets/classes/labels associate with each testing input datas


def create_training_dataset():
    global train_x
    global train_y

    for i in range(50):
        age = randint(13, 64)
        train_x.append(age)
        train_y.append(1)

        age = randint(65, 100)
        train_x.append(age)
        train_y.append(0)

    for i in range(950):
        age = randint(13, 64)
        train_x.append(age)
        train_y.append(0)

        age = randint(65, 100)
        train_x.append(age)
        train_y.append(1)


def create_testing_datasets():
    global test_x
    global test_y

    for i in range(5):
        age = randint(13, 64)
        test_x.append(age)
        test_y.append(1)

        age = randint(65, 100)
        test_x.append(age)
        test_y.append(0)

    for i in range(95):
        age = randint(13, 64)
        test_x.append(age)
        test_y.append(0)

        age = randint(65, 100)
        test_x.append(age)
        test_y.append(1)


def main():
    global train_x
    global train_y
    global test_x
    global test_y

    create_training_dataset()
    create_testing_datasets()

    # Convert python array to numpy array
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Shuffle the arrangement of the array
    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)
    # Normalise/Map age from 13-100 to 0-1 for Model.fit() process later on
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train_x = scaler.fit_transform(train_x.reshape(-1, 1))  # Formality reshape* for Model.fit() process later on
    scaled_test_x = scaler.fit_transform(test_x.reshape(-1, 1))

    # Design model layers
    model1 = Sequential([layers.Dense(units=16, input_shape=(1,), activation='relu', name='Input'),
                        layers.Dense(units=32, activation='relu', name='Middle', ),
                        layers.Dense(units=2, activation='softmax', name='Output')])

    model1.summary()

    # Compile the designed model with set parameters
    model1.compile(optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # ----- Start training+validate the compiled model using training dataset and print the accuracy result -----
    print('\nTraning Results:')
    # Validation split take a percentage of the training set out for validation, that is not included in the training
    # process. Validation set is used to compare with the training result to review/tune training parameters.
    # (overfitting if difference is big etc.)
    # Lower batch size contributes to better training result
    model1.fit(x=scaled_train_x, y=train_y, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)

    # ----- Predict the test data class/label (has or no side effect) using the trained model -----
    predictions = model1.predict(x=scaled_test_x, batch_size=10, verbose=0)
    rounded_predictions = np.argmax(predictions, axis=-1)

    print('\nPrediction results for 200 test sets:\nSide effect?\n\tNo(0)\t  Yes(1)\tPredicted Label')
    for ind, prediction in enumerate(predictions):
        print(prediction, '\t\t', end='')
        print(rounded_predictions[ind])

    pos = 0
    for j in rounded_predictions:
        if j == 1:
            pos += 1
    print('\nRate: ', "{:.2f}".format(pos/test_y.size*100), '%')  # Perfect is 50% ngam

    # ----- Check prediction accuracy/error with known text labels (test_y) using Confusion Matrix -----
    # Perfect prediction result of no error is
    # [[95  5]
    #  [ 5 95]]

    cm = confusion_matrix(y_true=test_y, y_pred=rounded_predictions)
    cm_plot_label = ['No side effect', 'Has side effect']  # Follow label/class index
    plot_confusion_matrix(cm=cm, classes=cm_plot_label, title='Confusion Matrix')


if __name__ == '__main__':
    main()
