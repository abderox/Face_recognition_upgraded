# @author abdelhadi mouzafir
import cv2
import numpy as np
import time
from face_recognition import facerecognition

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
i=0
def video(i):
    ret, frame = cap.read()

    if ret:
            cv2.namedWindow('Camera',cv2.WINDOW_NORMAL)
            cv2.imshow('Camera',frame)
            cv2.imwrite("folder/frame"+str(i)+".jpg", frame)
    else:
            print("No image detected. Please! try again")

while(cap.isOpened() and i <40):
    video(i)
    time.sleep(5)
    i+=1
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
    

cap.release()
cv2.destroyAllWindows()


facerecognition()