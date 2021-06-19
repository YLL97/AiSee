"""
A module to run and handle the TensorFlow Object Detection API. Modified from "Object Detection API Demo" by YL
"""

#!/usr/bin/env python
# coding: utf-8

# # Object Detection API Demo
# 
# <table align="left"><td>
#   <a target="_blank"  href="https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab
#   </a>
# </td><td>
#   <a target="_blank"  href="https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb">
#     <img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
# </td></table>

# Welcome to the [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image.

# > **Important**: This tutorial is to help you through the first step towards using [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to build models. If you just just need an off the shelf model that does the job, see the [TFHub object detection example](https://colab.sandbox.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb).

# # Setup

# Important: If you're running on a local machine, be sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md). This notebook includes only what's necessary to run in Colab.

import os
import pathlib
import sys

import tensorflow as tf
import numpy as np
import cv2
import six
from gui.audioio import MultithreadSpeak, MultithreadGetAudio
import time

# Import the object detection modules.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# change directory to tfod_api
root =  os.path.abspath(os.path.join(__file__ ,".."))
os.chdir(root + '/tfod_api')

# Patches:
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# Global Vars
CAP = cv2.VideoCapture(1, cv2.CAP_DSHOW)
CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
RESOLUTION = (1280, 720)
output_dict = None  # Global var to be used to store 'detection_boxes', 'detection_classes' and 'detection_scores' keys
announcer = MultithreadSpeak()
# Model preparation

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.
# output_dict = None # YL: Declare useful global variable to be use later

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)
    print(model_dir)
    model_dir = pathlib.Path(model_dir)/"saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# List of the strings that is used to add correct label for each box.

PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load an object detection model:

# C:\Users\user\.keras\datasets\ssd_mobilenet_v1_coco_2017_11_17
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# model_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
detection_model = load_model(model_name)

# Check the model's input signature, it expects a batch of 3-color images of type uint8:

print(detection_model.signatures['serving_default'].inputs)

# And returns several outputs:

# detection_model.signatures['serving_default'].output_dtypes
# detection_model.signatures['serving_default'].output_shapes

# Add a wrapper function to call the model, and cleanup the outputs:


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    # Python assumes that any name that is assigned to, anywhere within a function, is local to that function
    # unless explicitly told otherwise. If it is only reading from a name, and the name doesn't exist locally,
    # it will try to look up the name in any containing scopes (e.g. the module's global scope).
    global output_dict
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
               output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# Run it on each test image and show the results:


def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(   # Function Definition: https://github.com/tensorflow/models/blob/6a7b4d1aa96155588474aa14567c3bf975117223/official/vision/detection/utils/object_detection/visualization_utils.py#L535
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=0.5,
        line_thickness=4)

    return image_np


# YL: returns detected object and its percision
def get_classes_name_and_scores(
        boxes,
        classes,
        scores,
        category_index,  # category_index is a dictionary of id and classes
        max_boxes_to_draw=10,
        min_score_thresh=.5):  # returns bigger than 50% precision by default

    class_name_and_score_list = []

    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):  # Loop through each boxes(objects) in image
        if scores is None or scores[i] > min_score_thresh:   # if scores is higher than threshold default:0.5
            if classes[i] in six.viewkeys(category_index):   # if classes id exists in COCO class id
                class_name_and_score_dict = {
                    'name': category_index[classes[i]]['name'],
                    'score': '{}%'.format(int(100 * scores[i])),
                }
                class_name_and_score_list.append(class_name_and_score_dict)

    return class_name_and_score_list


def get_classes_name_only():
    obj_and_score_list = get_classes_name_and_scores(
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index)

    obj_list = []

    for dicts in obj_and_score_list:
        del dicts['score']  # remove 'score' key from each dictionary
        name_nested = list(
            dicts.values())  # somehow using list(dict.values()) function will return nested list weird
        name = name_nested[0]  # extract name from nested list [['name']] to list ['name']
        obj_list.append(name)

    return obj_list


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


class main():
    def __init__(self):
        self.quit = False
        self.imgpath = None
        self.image_ori = None
        self.image_labeled = None
        self.canscan = False
        self.scanned = False
        self.importmode = False
        announcer.speak('Welcome to General Object Detection Module')


    def import_infer(self):
        # image_np = cv2.imread("D:/Users/Leong/Pictures/road.jpg")
        image_np = cv2.imread(self.imgpath)
        self.image_ori = image_resize(image_np, height=720)
        self.image_labeled = show_inference(detection_model, image_np)

    def run_main(self):
        if self.importmode == False:  # Feed Mode
            ret, image_np = CAP.read()
            image_np = cv2.flip(image_np, 1)  # Flipped webcam input image
            img_labeled = show_inference(detection_model, image_np)
            cv2.imshow('General Object Detection', cv2.resize(img_labeled, RESOLUTION))
            get_ascii = cv2.waitKey(10)  # Set frame pause time (ms) ? *Put cv2.waitKey after imshow for immediated window property detection*
            detected_obj_list = get_classes_name_only()  # get list of detected object names

            if self.canscan:
                for name in detected_obj_list:
                    if (name != 'person'):
                        print(detected_obj_list)
                        announcer.speak(name)
                        self.canscan = False
                        return name
                    else:
                        announcer.speak("No Object Detected")
                        self.canscan = False
                        return "No Object Detected"
        else:  # Import Mode
            detected_obj_list = get_classes_name_only()  # get list of detected object names
            obj_dict = {}
            cv2.waitKey(10)  # else cv2.imshow will make program un-responding

            if self.scanned:
                cv2.imshow('General Object Detection', cv2.resize(self.image_labeled, RESOLUTION))
            else:
                cv2.imshow('General Object Detection', cv2.resize(self.image_ori, RESOLUTION))

            if self.canscan:  # ASCII for SPACE is 32:
                if  self.scanned == False:
                    cv2.imshow('General Object Detection', cv2.resize(self.image_labeled, RESOLUTION))
                    # Some magic to convert list of items into dict with label and amount
                    obj_dict = {i: detected_obj_list.count(i) for i in detected_obj_list}
                    if bool(obj_dict):  # Convert object to boolean to check if obj_dict is not empty
                        keys = list(obj_dict.keys())  # Extract keys from dict
                        vals = list(obj_dict.values())  # Extract values from dict
                        vals = [str(i) for i in vals]  # Shortcut to iterate vallist to convert to str
                        valkeys = list(map(str.__add__, vals, keys))  # join corresponding val elements to key elements
                        valkeysjoined = ', '.join(valkeys)  # Join all elements with ', ' to single element

                        announcer.speak(valkeysjoined)
                    else:
                        announcer.speak("No Object Detected")

                    self.canscan = False
                    self.scanned = True  # Lock
                    return obj_dict

        if cv2.getWindowProperty('General Object Detection', cv2.WND_PROP_AUTOSIZE) != 1.0:
            self.quit = True

        if self.quit:
            announcer.speak('Exiting Module 3')
            cv2.destroyAllWindows()

    def get_sigscan(self):  # signal is always True
        self.canscan = True

    def get_imgpath(self, imgpath):
        self.importmode = True
        self.scanned = False
        self.imgpath = imgpath
        announcer.speak("import")

    def get_sigclear(self):  # signal is always True
        self.importmode = False  # invert signal for application
        announcer.speak("clear")


if __name__ == "__main__":
    pass
