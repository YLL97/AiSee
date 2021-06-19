import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from confusion_matrix_plot import plot_confusion_matrix


def main():
    # Path to test datasets
    test1_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test1/'
    test2_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test2/'

    # Preprocess the data using ImageDataGenerator functions, based on MobileNet model input requirement
    test1_batch = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=test1_path, classes={'rm1':0,
                                       'rm5':1,
                                       'rm10':2,
                                       'rm20':3,
                                       'rm50':4,
                                       'rm100':5}, target_size=(224, 224), batch_size=10, shuffle=False)
    test2_batch = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
        directory=test2_path, classes={'rm1':0,
                                       'rm5':1,
                                       'rm10':2,
                                       'rm20':3,
                                       'rm50':4,
                                       'rm100':5}, target_size=(224, 224), batch_size=10, shuffle=False)

    # Load trained model
    model = load_model('D:/Users/Leong/Documents/FYP/bank_note_recognition/models/trained_model_v1.2.h5')

    # Prediction using test data (Testing)
    predictions = model.predict(x=test1_batch, verbose=0)
    test_labels = test1_batch.classes  # Retrieve the un-shuffled test1_batch classes in list
    print(test1_batch.class_indices)
    print(test_labels)

    # Show the confusion matrix
    cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
    cm_plot_label = ['RM1', 'RM5', 'RM10', 'RM20', 'RM50', 'RM100']
    plot_confusion_matrix(cm=cm, classes=cm_plot_label, title='Confusion Matrix')


if __name__ == '__main__':
    main()