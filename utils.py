import cv2
import dlib

BLUR_SIZE=300 #oddvalue to ensure symmetry
# Load the pre-trained facial landmark detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = detector(gray)
    return faces

#! image -> cv image
def detect_face_landmarks(face,image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    return landmarks
    
#!NOT COMPLETE
def blur_portion_of_image(image,x,y):
    pixel_region = image[y -BLUR_SIZE // 2:y + BLUR_SIZE // 2 + 1, x - BLUR_SIZE // 2:x + BLUR_SIZE // 2 + 1]
    blurred_pixel_region = cv2.GaussianBlur(pixel_region, (BLUR_SIZE, BLUR_SIZE))
    image[y, x] = blurred_pixel_region[BLUR_SIZE // 2, BLUR_SIZE // 2]
