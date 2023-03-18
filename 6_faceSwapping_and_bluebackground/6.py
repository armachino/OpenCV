from __future__ import print_function
import cv2 as cv
import numpy as np

bg_subtraction_methods=["KNN", "MOG2"]
bg_sub=bg_subtraction_methods[0]
## [create]
#create Background Subtractor objects

backSub = cv.createBackgroundSubtractorMOG2()

myColors = [
           ['Green',57,76,0,100,255,255],
           ]
colorValue = [
             [102,255,102],
             ]
my_points = []
## [create]

## [capture]
capture = cv.VideoCapture(0)


def contours(img):
    x, w, y, h = 0, 0, 0, 0
    contours, heirarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>500:
            #cv.drawContours(imgResult, cnt, -1, (0,255,0), 3)
            x, y, w, h = cv.boundingRect(cnt)
    return x+w//2, y

## [capture]
def findColor(img):
    count = 0
    new_points = []
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[1:4])
        upper = np.array(color[4:])
        mask = cv.inRange(imgHSV, lower, upper)
        x,y = contours(mask)
        cv.circle(imgResult, (x, y), 10, colorValue[count], cv.FILLED)
        if x!=0 and y!=0:
            new_points.append([x,y,count])
        count += 1   # To decide the Color
        #cv.imshow(color[0], mask)
    return new_points


while True:
    ret, frame = capture.read()
    cv.rectangle(frame, (0, 0), (100,100), (0,0,255),thickness=2)
    cv.rectangle(frame, (frame.shape[1]-100,0),(frame.shape[1],100), (0,255,0),thickness=2)

    if frame is None:
        break
    imgResult = frame.copy()
    
    new_point = findColor(frame)
    if len(new_point) != 0:
        print(new_point)
    # print(new_points)
    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    masked=cv.bitwise_and(frame, frame, mask=fgMask)
    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    cv.imshow('Frame masked', masked)
    ## [show]

    if cv.waitKey(1) & 0xFF==ord('d'):
            break
capture.release();
cv.destroyAllWindows();