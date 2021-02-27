"""Detecting face in video"""
import cv2
from random import randrange

#loading pretrained data
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#using webcam to capture realtime video
webcam  = cv2.VideoCapture(0)# '0' indicates that video is being captured from default cam i.e. webcam to play the recorded video enter the path for the video file

#iterate over the frames in the webcam unless webcam is closed
while True:
    #read current frame
    successfull_read_frame, frame = webcam.read()

    #converting frame into grayscale
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detecting face
    face_cordinates = trained_face_data.detectMultiScale(gray_scaled_img)

    for (x, y, w, h) in face_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        #to place a text above rectangle
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)#to change colors of the rectangle border

    #display current frame
    cv2.imshow("Detecting Face", frame)
    key =  cv2.waitKey(1)

    #to stop by pressing "Q" or "q"
    if key == 81 or key == 113: # ASCII value for Q=81 and for q=113
        break
#release the video capture object
webcam.release()

print("Code Complete")