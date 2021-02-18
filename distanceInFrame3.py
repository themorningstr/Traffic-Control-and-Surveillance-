import math
import argparse
import os
import cv2
import time
import numpy as np
import imutils
from PIL import Image

np.random.seed(42)

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", help = "yolo Weights/Config/Names folder", required=True)
ap.add_argument("-i", "--input", help = "path to input source", required=True)
ap.add_argument("-o", "--output", help = "output file name", required=True)
ap.add_argument("-l", "--lane", help = "Lane Number", default = 2)
args = vars(ap.parse_args())

lane = int(args["lane"])

# Class defining Point
class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def getValue_x(self):
		return self.x

	def getValue_y(self):
		return self.y


# Class defining Vector joining two points
class Line:
	def __init__(self, a, b):
		self.x = b.getValue_x() - a.getValue_x()
		self.y = b.getValue_y() - a.getValue_y()

	def getValue_x(self):
		return self.x

	def getValue_y(self):
		return self.y

	def dotProduct(self, v):
		return self.x*v.getValue_x() + self.y*v.getValue_y()
	

# Function for checking whether the point P is inside polygon ABCD
def isPointInsideFrame(P, A, B, C, D):
	AP = Line(A, P)
	AB = Line(A, B)
	AD = Line(A, D)
	costheta1 = AP.dotProduct(AB)/(AP.dotProduct(AP))**0.5
	costheta2 = AD.dotProduct(AB)/(AD.dotProduct(AD))**0.5

	BP = Line(B, P)
	BA = Line(B, A)
	BC = Line(B, C)
	costheta3 = BP.dotProduct(BA)/(BP.dotProduct(BP))**0.5
	costheta4 = BC.dotProduct(BA)/(BC.dotProduct(BC))**0.5
	
	DP = Line(D, P)
	DA = Line(D, A)
	DC = Line(D, C)
	costheta5 = DP.dotProduct(DA)/(DP.dotProduct(DP))**0.5
	costheta6 = DC.dotProduct(DA)/(DC.dotProduct(DC))**0.5

	return (costheta1>costheta2) and (costheta3>costheta4) and (costheta5>costheta6)


CONF = 0.5
THRES = 0.3

labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(4, 3), dtype="uint8")
LABEL_COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Loading pre-trained YOLOv3 model
weightsPath = os.path.sep.join([args["yolo"], "project-obj_best.weights"])
configPath = os.path.sep.join([args["yolo"], "project-obj.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get names of output layers, output for YOLOv4
def getOutputsNames(var):
    layersNames = var.getLayerNames()
    return [layersNames[i[0] - 1] for i in var.getUnconnectedOutLayers()]

# Initializing video writer to write the output video
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
# Counting total number of frames in the video
prop = cv2.CAP_PROP_FRAME_COUNT
total = int(vs.get(prop))

# Looping through each frame of the video
break_count = 0

while True:

	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# Converting image into a blob (N,C,H,W)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(getOutputsNames(net))		# Output from neural network
	end = time.time()

	boxes = []
	centers = []
	confidences = []
	IDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				centers.append([centerX, centerY])
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				IDs.append(classID)

	# Non-Maxima Suppression
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF, THRES)

	if break_count == 0:
		if lane == 1: 
			A = Point(165, 0)
			B = Point(279, 0)
			C = Point(451, 200)
			D = Point(140, 200)
		elif lane == 2:
			A = Point(354, 0)
			B = Point(458, 0)
			C = Point(630, 300)
			D = Point(324, 300)


	cv2.line(frame,(A.getValue_x(),0),(B.getValue_x(), 0),(0,0,255),2)

	cv2.line(frame,(B.getValue_x(),0),(C.getValue_x(), 300),(0,0,255),2)

	cv2.line(frame,(C.getValue_x(),300),(D.getValue_x(), 300),(0,0,255),2)

	cv2.line(frame,(D.getValue_x(),300),(A.getValue_x(), 0),(0,0,255),2)



	if len(indexes) > 0:
		for i in indexes:
			i = i[0]
			boxI = boxes[i]
			#(x,y) = (boxI[0], boxI[1])
			#(w,h) = (boxI[2], boxI[3])
			centeriX = boxI[0] + (boxI[2] // 2)
			centeriY = boxI[1] + (boxI[3] // 2)

			

			indexes_copy = list(indexes)
			indexes_copy.remove(i)

			for j in np.array(indexes_copy):
				j = j[0]
				boxJ = boxes[j]
				x,y = boxJ[0] , boxJ[1]
				w,h = boxJ[2] , boxJ[3]
				centerjX = boxJ[0] + (boxJ[2] // 2)
				centerjY = boxJ[1] + (boxJ[3] // 2)

				P = Point(x,y)

				#distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centerjX, 2))

				#if distance <= 300:
					#cv2.line(frame,(boxI[0] + (boxI[2] // 2),boxI[1] + (boxI[3] // 2)),(boxJ[0] + (boxJ[2] // 2), boxJ[1] + (boxJ[3] // 2)),(128,0,0),2)
				
				if isPointInsideFrame(P,A,B,C,D):
					distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centerjX, 2))

					if distance <= 300:
						cv2.line(frame,(boxI[0] + (boxI[2] // 2),boxI[1] + (boxI[3] // 2)),(boxJ[0] + (boxJ[2] // 2), boxJ[1] + (boxJ[3] // 2)),(128,0,0),2)
				else:
					continue

				cv2.rectangle(frame,(x,y),(int(x+w),int(y+h)),(0,255,0),1)
				text = "{}:{:.2f}".format(LABELS[IDs[j]],confidences[j])
				cv2.putText(frame,text,(int(x-5), int(y-5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1)



			#cv2.rectangle(frame,(x,y),(int(x+w),int(y+h)),(0,255,0),1)
			#text = "{}:{:.2f}".format(LABELS[IDs[i]],confidences[i])
			#cv2.putText(frame,text,(int(x-5), int(y-5)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1)
						
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
		if total > 0:
			timeLap = (end - start)
			print('Single frame took {:.2f} seconds'.format(timeLap))
			print("Estimated total time to finish: {:.2f}".format(timeLap * total))

	writer.write(frame)
	
	if break_count == -1:
		break
	else:
		break_count += 1

print('Cleaning up...')
writer.release()
vs.release()
