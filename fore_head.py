import cv2
import dlib

# Load the pre-trained facial landmark detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image = cv2.imread('path_to_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
detector = dlib.get_frontal_face_detector()
faces = detector(gray)

# Iterate over the detected faces
for face in faces:
    # Find the face landmarks
    landmarks = predictor(gray, face)

    # Iterate over the face landmarks and draw them on the image
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Display the image with face landmarks
cv2.imshow("Face Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
