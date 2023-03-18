import cv2
from scipy.spatial import distance as dist
import dlib  # for face and landmark detection

# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmark_predict = dlib.shape_predictor(
    './shape_predictor_68_face_landmarks.dat')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_smile.xml')

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# gray=25
# faces  = face_cascade.detectMultiScale(gray, 1.3, 5)


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # cv2.imshow('roi_color', roi_color)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy),
                          ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        if len(eyes) != 0:
            for (sx, sy, sw, sh) in eyes:
                cv2.rectangle(roi_color, (sx, sy),
                              ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Winked', (30, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
    return frame

    # faces = detector(img_gray)
    # for face in faces:
    #         # landmark detection
    #         shape = landmark_predict(img_gray, face)

    #         # converting the shape class directly
    #         # to a list of (x,y) coordinates
    #         shape = face_utils.shape_to_np(shape)

    #         # parsing the landmarks list to extract
    #         # lefteye and righteye landmarks--#
    #         lefteye = shape[L_start: L_end]
    #         righteye = shape[R_start:R_end]

    #         # Calculate the EAR
    #         left_EAR = calculate_EAR(lefteye)
    #         right_EAR = calculate_EAR(righteye)

    #         # Avg of left and right eye EAR
    #         avg = (left_EAR+right_EAR)/2
    #         if avg < blink_thresh:
    #             count_frame += 1  # incrementing the frame count
    #         else:
    #             if count_frame >= succ_frame:
    #                 cv2.putText(frame, 'Blink Detected', (30, 30),
    #                             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
    #             else:
    #                 count_frame = 0


video_capture = cv2.VideoCapture(0)
while video_capture.isOpened():
   # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calls the detect() function
    canvas = detect(gray, frame)

    # Displays the result on camera feed
    cv2.imshow('Video', canvas)

    # The control breaks once q key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done.
video_capture.release()
cv2.destroyAllWindows()
