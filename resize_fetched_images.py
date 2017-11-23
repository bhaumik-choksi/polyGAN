import os
import cv2

# Output dimension
dim = 28

files = os.listdir("images")
for file in files:
	img = cv2.imread("images/"+str(file), cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (dim,dim))
	cv2.imwrite("images/"+str(file), img)