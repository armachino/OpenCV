import cv2 
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

font = cv2.FONT_HERSHEY_SIMPLEX

cascPath = cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
eyePath = cv2.data.haarcascades+"haarcascade_eye.xml"
smilePath = cv2.data.haarcascades+"haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)


# # Load the image
# gray = cv2.imread('img/elich.jpg', 0)
# plt.figure(figsize=(12,8))
# plt.imshow(gray, cmap='gray')
# # plt.show()

# # Detect faces
# faces = faceCascade.detectMultiScale(
# gray,
# scaleFactor=1.1,
# minNeighbors=5,
# flags=cv2.CASCADE_SCALE_IMAGE
# )
# # For each face
# for (x, y, w, h) in faces: 
#     # Draw rectangle around the face
#     cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

# plt.figure(figsize=(12,8))
# plt.imshow(gray, cmap='gray')
# # plt.show()

# video_capture = cv2.VideoCapture(0)
# i=0
# while True:
#     i+=1
    
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#         )
#     if ret:
#         for (x, y, w, h) in faces:
#             print(i)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y:y+h, x:x+w]
#             smile = smileCascade.detectMultiScale(
#                 roi_gray,
#                 scaleFactor= 1.16,
#                 minNeighbors=35,
#                 minSize=(25, 25),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#             )
#             for (sx, sy, sw, sh) in smile:
#                 cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
#                 cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
#             eyes = eyeCascade.detectMultiScale(roi_gray)
#             for (ex,ey,ew,eh) in eyes:
#                 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#                 cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
#             faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.1,
#                 minNeighbors=5,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE
#                 )
#             cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
#             # Display the resulting frame
#         cv2.imshow('Video', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# video_capture.release()
# cv2.destroyAllWindows()

import numpy as np
gray = cv2.imread('img/elich.jpg', 0)
im = np.float32(gray) / 255.0
# Calculate gradient 
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
plt.figure(figsize=(12,8))
plt.imshow(mag)
plt.show()

face_detect = dlib.get_frontal_face_detector()
rects = face_detect(gray, 1)
for (i, rect) in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 255, 255), 3)
    
    
plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()


video_capture = cv2.VideoCapture(0)
flag = 0
overlay_face = cv2.imread('img/richi.jpg')
while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detect(gray, 1)
    if ret:
        for (i, rect) in enumerate(rects):

            (x, y, w, h) = face_utils.rect_to_bb(rect)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            dim=(w,h)
            resized_overlay_face = cv2.resize(overlay_face, dim, interpolation = cv2.INTER_AREA)
            frame[ y:y+h , x:x+w ] = resized_overlay_face

            cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()