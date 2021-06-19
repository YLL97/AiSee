"""
Ringgit Malaysia Bank Note Recognition Model Trainer Script
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from gui.audioio import MultithreadSpeak, MultithreadGetAudio
import threading
import time

root =  os.path.abspath(os.path.join(__file__ ,".."))
model = load_model(root + '/bank_note_recognition/models/trained_model_v1.2.h5')
CAP = cv2.VideoCapture(1, cv2.CAP_DSHOW)
CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
RESOLUTION = (1280, 720)
announcer = MultithreadSpeak()  # Create speaker object

threshold = 0.5


# Text Label Settings
LABELPOS = (10, 80)
LABELCOLOR = (255,255,255)
LABELSIZE = 3
LABELTHIC = 4

# Preprocessing of single image file for prediction
def prepare_image(file):
    img = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)  # !Keras loads image in RGB while OpenCV loads in BGR, convertion is need!
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST)
    # img = image.load_img(file, target_size=(224, 224))  # If input from a path use this line instead of above
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


class main:
    def __init__(self):
        self.class_list = ['RM1', 'RM5', 'RM10', 'RM20', 'RM50', 'RM100']  # Refer model identity
        self.class_list2 = ['RM 1', 'RM 5', 'RM 10', 'RM 20', 'RM 50', 'RM 100']  # For printing purpose
        self.imgpath = 'D:/Users/Leong/Documents/FYP/bank_note_recognition/dataset/test2/rm10/WIN_20210513_18_41_46_Pro.jpg'
        self.labelprint = ''
        self.canscan = False
        self.quit = False
        announcer.speak('Welcome to Bank Note Recognition Module')
        self.run_main()

    def run_main(self):
        # Read from webcam and display window
        ret, webcam_input = CAP.read()
        img = cv2.flip(webcam_input, 1)
        # img = cv2.putText(img, self.labelprint, LABELPOS, cv2.FONT_HERSHEY_SIMPLEX, LABELSIZE, LABELCOLOR, LABELTHIC, cv2.LINE_AA, False)
        cv2.imshow('Bank Note Recognition', cv2.resize(img, RESOLUTION))
        get_ascii = cv2.waitKey(10)  # Keep track of key press



        # Process input based on model input requirement
        processed_input = prepare_image(webcam_input)

        # Run prediction on obtain webcam image
        predictions = model.predict(x=processed_input, verbose=0)  # Return list of label scores
        label_max = predictions.argmax(axis=1)  # Return the index for max score
        score_max = np.max(predictions)  # Return the max score

        for ind, (label, label2) in enumerate(zip(self.class_list, self.class_list2)):
            if self.canscan:
                if label_max == ind:
                    if score_max >= threshold:
                        self.labelprint = label
                        print(f'Inference: {label}')
                        announcer.speak(f'{label}')

                        self.canscan = False
                        return label2
                    else:
                        print('No Bank Note Recognised')
                        announcer.speak('Try Again')

                        self.canscan = 0
                        return "Try Again"


        if cv2.getWindowProperty('Bank Note Recognition', cv2.WND_PROP_AUTOSIZE) != 1.0:
            self.quit = True

        if self.quit:
            announcer.speak('Exiting Module 2')
            cv2.destroyAllWindows()


    def get_sigscan(self):
        self.canscan = True


if __name__ == "__main__":
    pass
