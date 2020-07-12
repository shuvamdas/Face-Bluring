# USAGE
# python blur_face.py --image examples/adrian.jpg --face face_detector --method simple
# python blur_face.py --image examples/adrian.jpg --face face_detector --method pixelated

from blur.face_blurring import anonymize_face_pixelate
from blur.face_blurring import anonymize_face_simple
import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-m", "--method", type=str, default="simple",
	choices=["simple", "pixelated"],
	help="face blurring/anonymizing method")
ap.add_argument("-b", "--blocks", type=int, default=20,
	help="# of blocks for the pixelated blurring method")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)


image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
	
	confidence = detections[0, 0, i, 2]

	
	if confidence > args["confidence"]:
		
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		
		face = image[startY:endY, startX:endX]

		
		if args["method"] == "simple":
			face = anonymize_face_simple(face, factor=3.0)

		
		else:
			face = anonymize_face_pixelate(face,
				blocks=args["blocks"])

		image[startY:endY, startX:endX] = face


output = np.hstack([orig, image])
cv2.imshow("Output", output)
cv2.waitKey(0)
