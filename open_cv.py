import cv2
import dlib
import numpy as np

from utils import blur_portion_of_image, detect_face_landmarks, detect_faces

image_path = "input_imgs/1.jpeg"
image = cv2.imread(image_path)

def main():
    original_image = image.copy()
    faces = detect_faces(image)
    for each in faces:
        landmarks = detect_face_landmarks(each,image)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #blur_portion_of_image(image,x,y)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    
    concatenated_image = cv2.hconcat([image, original_image])
    # Display the concatenated image
    cv2.imshow('Combined Images', concatenated_image)
    cv2.waitKey(0)

main()
