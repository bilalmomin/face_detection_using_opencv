"""Detecting face in image"""
import cv2 

#loading pretrained data
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#loading image to detect face
img = cv2.imread('path//for//image.jpg')

#converting image into grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detecting face
face_cordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_cordinates)

#drawing rectangle around the face
"""this is how cordinates are used: cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
   in the printed face cordinates the first two points are x,y respectively and last two are width and height respectively
 where x2 = x+w 
       y2 = y+h """
# (x, y, w, h) = face_cordinates[0]
# cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
# cv2.rectangle(img, (75,23), (75+101,23+101), (0,255,0), 2)

# to detect a single face
# cv2.rectangle(img, face_cordinates, (0,255,0), 2)

#for multiple faces
# for enu,i in enumerate(face_cordinates):
#     cv2.rectangle(img, face_cordinates[enu], (0, 255, 0), 2)
#       OR
for (x, y, w, h) in face_cordinates:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    # cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)), 2)#to change colors of the rectangle border

#diplaying the image
cv2.imshow('Detecting Face', img)
cv2.waitKey()


print("Code Complete")