"""
Webcam port test script
"""

import cv2


def main():
    camtest(1)


def camsearch():
    cams_test = 10
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        print("i : " + str(i) + " /// result: " + str(test))


def camtest(cam_num):
    cap = cv2.VideoCapture(cam_num)
    while True:
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('frame', cv2.resize(frame, (720, 480)))
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
