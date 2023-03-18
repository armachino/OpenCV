import numpy as np
import cv2 as cv

# ANACONDA_PACKAGE="/home/arman/anaconda3/envs/openCV/lib/python3.10/site-packages/cv/data/"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_eye.xml')

overlay_face = 'img/elich.jpg'
# load the overlay image. size should be smaller than video frame size
overlay_face = cv.imread('img/richi.jpg')

# Get Image dimensions
img_height, img_width, _ = overlay_face.shape

# face_cascade = cv.CascadeClassifier(ANACONDA_PACKAGE+'haarcascade_frontalface_default.xml')
# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv.imread('./img/pro.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    # frame_resized = rescaleFrame(frame)
    # frame_resized = changeRes(frame.shape[1]*scale)
    
    # cv.imshow('Video',frame)
    if isTrue:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            # img = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            dim=(w,h+20)
            resized_overlay_face = cv.resize(overlay_face, dim, interpolation = cv.INTER_AREA)
            frame[ y:y+h+20 , x:x+w ] = resized_overlay_face

            # frame[ y:y+img_height , x:x+img_width ] = overlay_face
            # roi_gray = gray[y:y+h, x:x+w]
            # roi_color = frame[y:y+h, x:x+w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv.imshow('img',frame)



    if cv.waitKey(1) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()


# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
