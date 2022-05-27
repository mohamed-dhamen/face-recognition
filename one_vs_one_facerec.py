
import face_recognition
import cv2
from simple_facerec import SimpleFacerec
#Face encoding first image
img = cv2.imread("Messi.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]
#Face encoding second image
img2 = cv2.imread("Messi.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

#Comparison of images
result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

#Encode all face in the dataset