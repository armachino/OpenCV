# import numpy as np
# import cv2
# import time

# video = cv2.VideoCapture(0)
# previous_frame = None
# start_time = time.time()
# blur = 25

# frames = []
# i=0
# started=False
# d=0
# while True:
#     ret, frame = video.read()
#     if time.time() - start_time <2:
#         print(time.time() - start_time)
#         # if not started:
#         #     frames=frame
#         # else:
#         # frames+=frame
#         frames.append(frame)
#         i+=1
#         started=True
#     else:
#         # if d==0:
#         #     frames=frames/i
#         avg_img = np.mean(frames, axis=0)
#         avg_img = avg_img.astype(np.uint8)
#         # avg_img=cv2.GaussianBlur(avg_img, (blur, blur), 0)
#         cv2.imshow("avg_img", avg_img)
#         msk=np.logical_and(avg_img,frame)
#         frame=cv2.bitwise_and(frame, frame)
#         cv2.imshow("Foreground", frame)
#         d=1

#     # if ret == True:

#         # Use the q button to quit the operation
#     if cv2.waitKey(60) & 0xff == ord('q'):
#         break
