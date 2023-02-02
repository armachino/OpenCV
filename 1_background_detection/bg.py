from __future__ import print_function
import cv2 as cv

bg_subtraction_methods=["KNN", "MOG2"]
bg_sub=bg_subtraction_methods[1]
## [create]
#create Background Subtractor objects
if bg_sub == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
capture = cv.VideoCapture(0)

## [capture]

while True:
    ret, frame = capture.read()
    if frame is None:
        break

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