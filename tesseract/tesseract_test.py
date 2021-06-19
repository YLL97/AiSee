import pytesseract
import cv2
import numpy as np


def preprocessing(image):
    ratio = 70
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # img = cv2.GaussianBlur(img, (7, 7),0)
    cv2.imshow('haha', cv2.resize(img, (int(1280 * ratio / 100), int(720 * ratio / 100)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    # img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)
    # cv2.imshow('grey', cv2.resize(img2, (int(1280 * ratio/100), int(720 * ratio/100)), interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)

    return img

# Function to check if two rectangles overlaps
def rect_overlap(x1,y1,x2,y2,x3,y3,x4,y4):
    if (x1 >= x4 or x3 >= x2):
        return False
    if (y1 >= y4 or y3 >= y2):
        return False
    return True


def tesseract_wrapper(image, mode):
    width = int(image.shape[1])
    height = int(image.shape[0])
    boxlist = []
    textlist = []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV default mode is in GBR, tesseract uses RGB

    # tesseract imagename outputbase [-l lang] [--oem ocrenginemode] [--psm pagesegmode] [configfiles...]
    config = r'--oem 3 --psm 1 outputbase digits'
    raw_result = pytesseract.image_to_data(image, config=config)
    print(raw_result)

    for idx, b in enumerate(raw_result.splitlines()):
        if idx != 0:
            b = b.split()
            print(idx, '\t', b)
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                if mode == 0:
                    boxlist.append([x, y, x + w, y + h])
                    textlist.append(b[11])
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(image, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (50, 50, 255), 2)

                if mode == 1:
                    if not (w == width and h == height):  # Reject any glitchy full frame detection
                        boxlist.append([x, y, x + w, y + h])
                        textlist.append(b[11])

    if mode == 0:
        return image, textlist

    if mode == 1:
        img2 = image
        newtextlist = []
        # --- Algorithm to only print non overlapping boxes ---
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
                cv2.putText(image, textlist[idx], (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (50, 50, 255), 2)
                newtextlist.append(textlist[idx])
                print('\tok')

        return image, newtextlist


def main():
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
    Path = 'D:/Users/Leong/Pictures/Productinfo.jpg'
    Path2 = 'D:/Users/Leong/Pictures/Ingredients.jpg'

    img = cv2.imread(Path)
    img = preprocessing(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV default mode is in GBR, tesseract uses RGB

    img2 = cv2.imread(Path2)

    img1, textl1 = tesseract_wrapper(img.copy(), 0)
    img2, text12 = tesseract_wrapper(img.copy(), 1)  # Wanna see detail code go see fucntion la


    print(img1 is img2)
    print(textl1 is text12)

    # Scaling Leong 23/1/2021
    scale_percent = 70

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    resized1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('image1', resized1)
    cv2.imshow('image2', resized2)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()