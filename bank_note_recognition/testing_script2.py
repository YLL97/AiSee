import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from confusion_matrix_plot import plot_confusion_matrix
import cv2

# Preprocess the data without using ImageDataGenerator
def prepare_image(rm, file, loadmode=0):
    test1_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test1/'
    test2_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test2/'

    try:
        if loadmode == 0:  # load via Keras built-in approach
            img = image.load_img(os.path.join(test2_path, rm, file),
                                 target_size=(224, 224))
        if loadmode == 1:
            img = cv2.imread(os.path.join(test2_path, rm, file))  # load via OpenCV approach
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # !Keras loads image in RGB while OpenCV loads in BGR, convertion is need!
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)

        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    except:
        print('Invalid loadmode entered: 0 - load via tensorflow.keras.preprocessing.image.load_img, 1 - load via OpenCv imread()')


def main():
    # Path to test datasets
    test1_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test1/'
    test2_path = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test2/'

    # Create list of numpy image tensors
    rmlist = ['rm1', 'rm5', 'rm10', 'rm20', 'rm50', 'rm100']
    dirlists = []
    processed_test2_batch = []

    for rm in rmlist:
        dirlists.append(os.listdir(os.path.join(test2_path, f'{rm}')))

    for rm, list in zip(rmlist, dirlists):  # Parallel looping
        templist = []
        for file in list:
            templist.append(prepare_image(rm, file, 1))
        processed_test2_batch.append(templist)

    # Check dimension
    print(len(processed_test2_batch))
    print(len(processed_test2_batch[0]))
    print(len(processed_test2_batch[0][0]))
    print(len(processed_test2_batch[0][0][0]))
    print(len(processed_test2_batch[0][0][0][0]))
    print(len(processed_test2_batch[0][0][0][0][0]))

    # load trained model
    model = load_model('D:/Users/Leong/Documents/FYP/bank_note_recognition/models/trained_model_v1.2.h5')

    # Prediction using test data (Testing)
    # Create numpy arrays of dimension [90, 1] as input for model.predict
    predictions = np.empty((0,6), dtype='float32')  # Define motherfucker numpy space first!
    for i in range(6):
        for j in range(15):
            prediction = model.predict(x=processed_test2_batch[i][j], verbose=0)
            predictions = np.vstack([predictions, prediction[0]])

    # Define the test labels (Refer training_script.py to review model class labels)
    test_labels = np.empty((0), dtype='int32')  # Empty np array
    for k in range(6):
        for _ in range(15):
            test_labels = np.append(test_labels, k)

    print(test_labels)

    # Show the confusion matrix
    cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
    cm_plot_label = ['RM1', 'RM5', 'RM10', 'RM20', 'RM50', 'RM100']
    plot_confusion_matrix(cm=cm, classes=cm_plot_label, title='Confusion Matrix')


if __name__ == '__main__':
    main()