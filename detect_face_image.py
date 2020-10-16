import cv2
import numpy as np
import argparse

# Get user supplied values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(args["image"])
cv2.imshow("Original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cv2.imshow("Gray",gray)

gray1 = np.copy(gray)

#eq = cv2.equalizeHist(gray1)
#cv2.imshow("eq", eq)

#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#eq = clahe.apply(gray1)
#cv2.imshow("eq", eq)


blurred = cv2.GaussianBlur(gray1, (5, 5), 0)

#cv2.imshow("Blured", blurred)

eroded = cv2.erode(gray1.copy(), None, iterations=5)
#cv2.imshow("erode", eroded)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    eroded,
    scaleFactor = 1.05,
    minNeighbors = 5,
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

height, width, channels = image.shape
print(height)
print(width)
print(channels)

cv2.imwrite("output.jpg", image)

cv2.imshow("{0} Faces found".format(len(faces)), image)
cv2.waitKey(0)
