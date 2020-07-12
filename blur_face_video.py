# USAGE
# python blur_face_video.py --face face_detector --method simple
# python blur_face_video.py --face face_detector --method pixelated

from pyimagesearch.face_blurring import anonymize_face_pixelate
from pyimagesearch.face_blurring import anonymize_face_simple
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

ap = argparse.ArgumentParser()
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

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		
		confidence = detections[0, 0, i, 2]

		
		if confidence > args["confidence"]:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			
			face = frame[startY:endY, startX:endX]

			
			if args["method"] == "simple":
				face = anonymize_face_simple(face, factor=3.0)

			
			else:
				face = anonymize_face_pixelate(face,
					blocks=args["blocks"])

			frame[startY:endY, startX:endX] = face

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
