import numpy as np
import cv2
 
#Initializing the face and eye cascade classifiers from xml files
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
 
#Variable store execution state
first_read = True
 
#Starting the video capture
cap = cv2.VideoCapture(0)
# ret,img = cap.read()
 
while(True):
    ret,img = cap.read()
    #Converting the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray,5,1,1)
 
    #Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
 
            #roi_face is face which is input to eye classifier
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
 
            #Examining the length of eyes object for eyes
            if(len(eyes)>=2):
                #Check if program is running for detection
                if(first_read):
                    cv2.putText(img,
                    "Eye detected press s to begin",
                    (70,70), 
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0,255,0),2)
                else:
                    cv2.putText(img,
                    "Eyes open!", (70,70),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (255,255,255),2)
            else:
                if(first_read):
                    #To ensure if the eyes are present before starting
                    cv2.putText(img,
                    "No eyes detected", (70,70),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0,0,255),2)
                else:
                    #This will print on console and restart the algorithm
                    print("Blink detected--------------")
                    cv2.waitKey(3000)
                    first_read=True
             
    else:
        cv2.putText(img,
        "No face detected",(100,100),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (0,255,0),2)
 
    #Controlling the algorithm with keys
    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        #This will start the detection
        first_read = False
 
cap.release()
cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import dlib
# from scipy.spatial import distance as dist


# JAWLINE_POINTS = list(range(0, 17))
# RIGHT_EYEBROW_POINTS = list(range(17, 22))
# LEFT_EYEBROW_POINTS = list(range(22, 27))
# NOSE_POINTS = list(range(27, 36))
# RIGHT_EYE_POINTS = list(range(36, 42))
# LEFT_EYE_POINTS = list(range(42, 48))
# MOUTH_OUTLINE_POINTS = list(range(48, 61))
# MOUTH_INNER_POINTS = list(range(61, 68))
# EYE_AR_THRESH = 0.22
# EYE_AR_CONSEC_FRAMES = 3
# EAR_AVG = 0
# COUNTER = 0
# TOTAL = 0

# def eye_aspect_ratio(eye):
#     # compute the euclidean distance between the vertical eye landmarks
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     # compute the euclidean distance between the horizontal eye landmarks
#     C = dist.euclidean(eye[0], eye[3])
#     # compute the EAR
#     ear = (A + B) / (2 * C)
#     return ear

# # to detect the facial region
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# # capture video from live video stream
# cap = cv2.VideoCapture(0)
# while True:
#     # get the frame
#     ret, frame = cap.read()
#     #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#     if ret:
#         # convert the frame to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rects = detector(gray, 0)
#         for rect in rects:
#             x = rect.left()
#             y = rect.top()
#             x1 = rect.right()
#             y1 = rect.bottom()
#             # get the facial landmarks
#             landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
#             # get the left eye landmarks
#             left_eye = landmarks[LEFT_EYE_POINTS]
#             # get the right eye landmarks
#             right_eye = landmarks[RIGHT_EYE_POINTS]
#             # draw contours on the eyes
#             left_eye_hull = cv2.convexHull(left_eye)
#             right_eye_hull = cv2.convexHull(right_eye)
#             # cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1) # (image, [contour], all_contours, color, thickness)
#             # cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
#             # compute the EAR for the left eye
#             ear_left = eye_aspect_ratio(left_eye)
#             # compute the EAR for the right eye
#             ear_right = eye_aspect_ratio(right_eye)
#             # compute the average EAR
#             ear_avg = (ear_left + ear_right) / 2.0
#             # detect the eye blink
#             if ear_avg < EYE_AR_THRESH:
#                 COUNTER += 1
#             else:
#                 if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                     TOTAL += 1
#                     print("Eye blinked : ",TOTAL," time")
#                     cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0),1)  # (image, [contour], all_contours, color, thickness)
#                     cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
#                 COUNTER = 0

#             cv2.putText(frame, "Blinks{}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 1)
#             cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,0,0), 1)
#         cv2.imshow("Winks Found", frame)
#         key = cv2.waitKey(1) & 0xFF
#         # When key 'Q' is pressed, exit
#         if key is ord('q'):
#             break

# # release all resources
# cap.release()
# # destroy all windows
# cv2.destroyAllWindows()






# import argparse
# import time
# import cv2
# import dlib
# import imutils
# import numpy as np
# from imutils import face_utils
# from imutils.video import FileVideoStream, VideoStream
# from scipy.spatial import distance as dist
# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear
# ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-p", "--shape-predictor", required=True, help="path to facial landmark predictor"
# )
# ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
# args = vars(ap.parse_args())
# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 3
# COUNTER = 0
# TOTAL = 0
# print("[INFO] loading facial landmark predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(args["shape_predictor"])
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
# vs = VideoStream(src=0).start()
# fileStream = False
# time.sleep(1.0)
# while True:
#     if fileStream and not vs.more():
#         break
#     frame = vs.read()
#     frame = imutils.resize(frame, width=800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)
#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#         else:
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 TOTAL += 1
#             COUNTER = 0
#         cv2.putText(
#             frame,
#             "Blinks: {}".format(TOTAL),
#             (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 0, 255),
#             2,
#         )
#         cv2.putText(
#             frame,
#             "EAR: {:.2f}".format(ear),
#             (300, 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 0, 255),
#             2,
#         )
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break
# cv2.destroyAllWindows()
# vs.stop()
