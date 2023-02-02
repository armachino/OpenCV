
import cv2
import numpy as np

# yellow #purple #gren
myColors = [
        #     ['Yellow',18,40,90,30,255,255],          # HSV min and max values for Color Detection
        #    ['Purple',110,46,50,140,255,255],
           ['Green',57,76,0,100,255,255],
        #    ['Blue',90,48,0,118,255,255]
           ]
colorValue = [
            # [0,255,255],                   #BGR Code  ['Blue',90,48,0,118,255,255]
            #   [255,0,127],
             [102,255,102],
            #  [76,0,11]
             ]
my_points = []   #[x, y, ColorID]


def drawCanvas(points):
    for point in points:
        cv2.circle(imgResult, (point[0], point[1]), 10, colorValue[point[2]], cv2.FILLED)

def contours(img):
    x, w, y, h = 0, 0, 0, 0
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            #cv2.drawContours(imgResult, cnt, -1, (0,255,0), 3)
            x, y, w, h = cv2.boundingRect(cnt)
    return x+w//2, y

def findColor(img):
    count = 0
    new_points = []
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[1:4])
        upper = np.array(color[4:])
        mask = cv2.inRange(imgHSV, lower, upper)
        x,y = contours(mask)
        cv2.circle(imgResult, (x, y), 10, colorValue[count], cv2.FILLED)
        if x!=0 and y!=0:
            new_points.append([x,y,count])
        count += 1   # To decide the Color
        #cv2.imshow(color[0], mask)
    return new_points

# using a webcam
cam = cv2.VideoCapture(0)
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))
cam.set(3,640)
cam.set(4,480)
cam.set(10,130)
#cam.set(10, 130)  # brightness
while True:
    success, img = cam.read()
    if success == True:
        imgResult = img.copy()
        new_points = findColor(img)
        if len(new_points) != 0:
            for i in new_points:
                my_points.append(i)
        if len(my_points)!=0:
            drawCanvas(my_points)
        imgResult = cv2.flip(imgResult,1)
        cv2.imshow("Result", imgResult)
        out.write(imgResult)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            my_points = []
            break
cam.release()
out.release()
cv2.destroyAllWindows()