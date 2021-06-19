import os
import pytesseract
import cv2
import imutils
from tesseract.transform import four_point_transform
from gui.audioio import MultithreadSpeak, MultithreadGetAudio
import numpy as np
import time

# Link pytesseract wrapper to the installed Tesseract engine
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
# tesseract imagename outputbase [-l lang] [--oem ocrenginemode] [--psm pagesegmode] [configfiles...]
CONFIG = r'--oem 3 --psm 1 outputbase digits'
CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)
CAP.set(3, 1920)
CAP.set(4, 1080)
RESOLUTION = (1280, 720)
RESOLUTION2 = (720, 1280)
RESOLUTION3 = (509, 678)
thresh_val = 180  # Ori is 180  # If using Otsu thresholding, this var is automatically determined implicitly
announcer = MultithreadSpeak()
# Ref: https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale

def preprocess_threshold(image, thresh_val=180, adaptive=False):
    """
    Function to do simple thresholding
    :param image: image np_array
    :param thresh_val: threshold value
    :param adaptive: bool, True for adaptive thresholding, else normal thresholding
    :return: The threshold-ed image
    """

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if adaptive == False:
        # ret, img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # cv2.imshow('Optical Character Recognition2', cv2.resize(img, RESOLUTION))
    return img


def preprocess_fpt(image, admin=0):
    """
    Find largest rectangular contour and implement Four-Point Transformation
    :param image:  image np_array
    :param admin: 0 = not prompting windows, 1 = admin mode: prompt windows for each process
    :return: The transformed image
    """
    # --- Load image and preprocess it using filters ---
    orig = image.copy()

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gblur, 30, 100)

    if admin == 1: cv2.imshow("Edged", cv2.resize(edged, RESOLUTION)); cv2.waitKey(0)
    print("STEP 1: Edge Detection Completed")

    # --- Find the contours in the edged image, keeping only the largest ones ---
    # Return structure is a tuple with 2 elements: ([arr(1,c1,1,2), arr(1,c2,1,2), ..., arr(1,cn,1,2)], arr(1,1,h,4)),
    # where c1, c2, ...etc. is the number of points for that particular contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # grab_contours is just a wrapper function with statement cnts = cnts[0], grabbing the list of contours only,
    # ignoring the second element of np arr representing hierarchy, that nobody knows what does it means. By YL
    cnts = imutils.grab_contours(cnts)
    # cnts = [arr(1,c1,1,2), arr(1,c2,1,2), ..., arr(1,cn,1,2)] up until this point
    # Arrange the list of contour np arrays based on its area, in descending order, and only taking the first 5 largest contours
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # Algorithm to select the contour that approximate to a rectangular shape for perspective transformation
    # loop over the 5 selected contours
    for c in cnts:
        peri = cv2.arcLength(c, True)  # Function computes a curve length or a closed contour perimeter. Ref: OpenCV doc
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Ref: Ramer–Douglas–Peucker algorithm
        # if our approximated contour has four points, then we can assume that we have found our boundary
        if len(approx) == 4:
            planeCnt = approx
            # show the contour (outline) of the piece of paper
            cv2.drawContours(image, [planeCnt], -1, (0, 255, 0), 8)
            if admin == 1: cv2.imshow("Outline", cv2.resize(image, RESOLUTION)); cv2.waitKey(0)
            print("STEP 2: Find contours of paper completed")

            # --- Apply four point transform to obtain a top-down view ---
            warped = four_point_transform(orig, planeCnt.reshape(4, 2))

            if admin == 1: cv2.imshow("Scanned", warped); cv2.waitKey(0)
            print("STEP 3: Apply perspective transform completed")
            print("preprocess_fpt Completed")
            return warped

    return np.array(0)  # Return dummy numpy array if fpt is not successful for comparing purpose later

def rect_overlap(x1,y1,x2,y2,x3,y3,x4,y4):
    if (x1 >= x4 or x3 >= x2):
        return False
    if (y1 >= y4 or y3 >= y2):
        return False
    return True


def tesseract_wrapper(image, mode):
    '''
    Detect texts using Tesseact OCR with set CONFIG and draw detection boxes to be shown
    :param image: numpy image to be OCR
    :param mode: 0 - normal, 1 - eliminates overlapping detection boxes
    :return: numpy image, detected texts (str)
    '''
    width = int(image.shape[1])
    height = int(image.shape[0])
    boxlist = []
    textlist = []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV default mode is in GBR, tesseract uses RGB

    raw_result = pytesseract.image_to_data(image, config=CONFIG)
    # print(raw_result)

    for idx, b in enumerate(raw_result.splitlines()):
        if idx != 0:
            b = b.split()
            # print(idx, '\t', b)
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                if mode == 0:
                    boxlist.append([x, y, x + w, y + h])
                    textlist.append(b[11])
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                                (255, 0, 0), 2)

                if mode == 1:
                    if not (w == width and h == height):
                        boxlist.append([x, y, x + w, y + h])
                        textlist.append(b[11])

    if mode == 0:
        return image, textlist

    if mode == 1:
        img2 = image
        newtextlist = []
        # --- Algorithm to only print non overlapping boxes ---
        # Reject any glitchy full frame detection
        for idx, box in enumerate(boxlist):
            print(idx)
            overlap = False
            for i in range(len(boxlist)):
                if i == idx:
                    continue
                overlap = rect_overlap(box[0], box[1], box[2], box[3], boxlist[i][0],
                                       boxlist[i][1], boxlist[i][2],
                                       boxlist[i][3])
                if overlap:
                    print('\tdeng')
                    break
            if not overlap:
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                cv2.putText(image, textlist[idx], (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.75,
                            (50, 50, 255), 2)
                newtextlist.append(textlist[idx])
                print('\tok')

        return image, newtextlist

class main():
    def __init__(self):
        self.docmode = False
        self.canscan = False
        self.imgpath = None
        self.scanned = False
        self.sigclear = False
        self.quit = False
        self.text_detected = False
        self.image_ori = None
        self.image_labeled = None
        self.mem = True
        announcer.speak('Welcome to OCR Module')

    # imgpath = 'D:/Users/Leong/Pictures/ChessC.jpg'
    # pcimg = cv2.imread(imgpath)
    def run_main(self, mode):
        if mode==1:  # Webcam Mode
            ret, self.image_ori = CAP.read()  # Webcam self.image_ori
        elif mode==2:  # Import Mode
            if self.imgpath != '':  # Check if the imgpath Queue has been read before
                self.image_ori = cv2.imread(self.imgpath)
            else:
                pass

        if self.scanned and self.text_detected:
            cv2.imshow('Optical Character Recognition', cv2.resize(cv2.cvtColor((self.image_labeled), cv2.COLOR_RGB2BGR), RESOLUTION))
        else:  # Refer main
            if self.sigclear == False:  # If the current preview state has not been cleared
                cv2.imshow('Optical Character Recognition', cv2.resize(self.image_ori, RESOLUTION))
            else:
                self.scanned = False
                self.sigclear = False

        if self.canscan:  # ASCII for SPACE is 32:
            announcer.speak("Captured")
            if self.scanned == False:
                if self.docmode == False:  # Normal mode (no 4p transform)
                    if mode == 1:
                        img = preprocess_threshold(self.image_ori, thresh_val, False)
                    elif mode == 2:
                        img = preprocess_threshold(self.image_ori, thresh_val, False)
    
                else: # Document mode (4p transform)
                    img = preprocess_fpt(self.image_ori, 0)  # 0 = normal, 1 = admin mode
                    if img.any() == True:  # If img is a numpy array *If any of the element in numpy array (image) is True (not 0)*
                        if mode == 1:
                            img = preprocess_threshold(img, thresh_val, False)
                        elif mode == 2:
                            img = preprocess_threshold(img, thresh_val, False)
                        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        if mode == 1:
                            img = preprocess_threshold(self.image_ori, thresh_val, False)
                        elif mode == 2:
                            img = preprocess_threshold(self.image_ori, thresh_val, False)
                        print("No Document Detected, Scanning in normal mode instead")
    
                self.image_labeled, textlist = tesseract_wrapper(img.copy(), 0)  # If scanmode not equals to 0 or 1 error will raised

                seperator = ' '
                textlist = seperator.join(textlist)
                print(textlist)
                time.sleep(2.5)
                if bool(textlist):  # Check if there is text
                    announcer.winspeak(textlist)
                    self.canscan = False
                    self.scanned = True
                    self.text_detected = True
                    return textlist
                else:
                    announcer.speak('No Text Detected!')
                    self.canscan = False
                    self.scanned = False
                    self.text_detected = False
                    return 'No Text Detected!'
            self.canscan = False  # FIXME temporary fix for bug

        cv2.waitKey(10)  # Add delay to avoid crash while (slowdown looping)
        if cv2.getWindowProperty('Optical Character Recognition', cv2.WND_PROP_AUTOSIZE) != 1.0:
            self.quit = True

        if self.quit:
            announcer.speak('Exiting Module 1')
            announcer.winspeak_stop()
            cv2.destroyAllWindows()

    def get_sigscan(self):
        self.canscan = True

    def get_docmode(self, ticked):  # Checkbox
        self.docmode = ticked
        if self.mem == True:
            announcer.speak("Document Mode")
            self.mem = False
        else:
            announcer.speak("Normal Mode")
            self.mem = True

    def get_imgpath(self, imgpath):
        self.imgpath = imgpath
        announcer.speak("import")

    def get_sigclear(self):
        self.scanned = True  # invert signal for application
        announcer.speak("clear")
        announcer.winspeak_stop()


if __name__ == "__main__":
    pass
