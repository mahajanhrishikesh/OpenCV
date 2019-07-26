#importing the required libraries
import cv2
import numpy as np
import keyboard
import time

def calculateAngle(far, start, end):
    """Cosine rule"""
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def countFingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        cnt = 0
        if type(defects) != type(None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s, 0])
                end = tuple(contour[e, 0])
                far = tuple(contour[f, 0])
                angle = calculateAngle(far, start, end)
                
                # Ignore the defects which are small and wide
                # Probably not fingers
                if d > 10000 and angle <= math.pi/2:
                    cnt += 1
        return True, cnt
    return False, 0

cap = cv2.VideoCapture(0)   #Starts camera

while True:
    
    _,frame = cap.read()
    
    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image = frame[100:300, 100:300]
    #cv2.rectangle(frame,(roi_left,roi_top),(roi_right,roi_bottom),(0,255,0),2)
    hsv = cv2.cvtColor(cv2.medianBlur(crop_image,15), cv2.COLOR_BGR2HSV)
    
    low = np.array([0,10,20])
    high = np.array([30,255,255])
    
    #applying mask 
    mask = cv2.inRange(hsv,low,high)
    #Noise elimination
    mask = cv2.dilate(mask,None,iterations = 2)
    
    #extracting contours from the region of interest
    cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>0:
        for c in cnts:
            if cv2.contourArea(c) > 4000:
                hull = cv2.convexHull(c)
                #cv2.drawContours(crop_image,[hull,0,(0,0,255),2])
                ret,fc = countFingers(c)
                if ret == True:
                    if fc == 0:
                        keyboard.press(" ")
                        time.sleep(0.001)
    
    cv2.imshow("Video", frame)
    cv2.imshow("Hand",mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()