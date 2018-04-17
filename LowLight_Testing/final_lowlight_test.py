
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import numpy as np
from imutils import face_utils
import argparse
import datetime
import imutils
import dlib
import cv2

# Input the scaling ratio that will determine the size of the frames during processing
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--scale-ratio", required=False,
	help="Ratio at which to scale the input video frames")
args = vars(ap.parse_args())

# Initialising the face and facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the text file for recording test results
fullTestResults = open("fullTestResults.txt", "w+")
vidNum = 1
fullREyeSimil = 0
fullLEyeSimil = 0
fullMSimil = 0

while vidNum < 2:
	
	# Provide the path to the video and the annotation files
	if (vidNum < 10):
		annotPath = ("/home/pi/FYP/luxSensor/luxVideos/luxLevel-5-Points")
		videoPath = ("/home/pi/FYP/luxSensor/luxVideos/luxLevel-5.avi")
		testFile = ("TestResults-00" + str(vidNum) + ".txt")
	
	# Open the video sample
	print("Running fatigue detection algorithm...")
	vc = cv2.VideoCapture(videoPath)
	# Open the test results file for the individual video sample
	testResults = open(testFile,"w+")
	frameCount = 1
	finalREye = 0
	finalLEye = 0
	finalMouth = 0
	scalingRatio = float(args["scale_ratio"])
	
	# Repeat loop while frames available for processing
	while True:
		
		ret, frame = vc.read()
		if not ret:
			print("Capture Failed")
			break
		# Find the height and width of input video and scale the frame using scaling ratio inputted by user
		width = vc.get(3)
		height = vc.get(4)
		scaledWidth = int(width/scalingRatio)
		scaledHeight = int(height/scalingRatio)
		frame = cv2.resize(frame, (scaledWidth, scaledHeight))
		
		# Begin the image processing on frame
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(8,8))
		gray = clahe.apply(gray)
		gray = cv2.medianBlur(gray,13)

		imgframe = Image.fromarray(gray)
		enhancer_object = ImageEnhance.Contrast(imgframe)
		out = enhancer_object.enhance(500)
		gray = np.array(out)
		
		# Run face detector on frame
		faces = detector(gray, 0)
	
		#Default values if no face found within frame
		right_eye_curr = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
		left_eye_curr = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
		mouth_curr = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
		
		# Run the facial landmark detector on the face region returned by previous face detector
		for face in faces:
			face_shape_curr = predictor(gray, face)
			face_shape_curr = face_utils.shape_to_np(face_shape_curr)
			
			# Set the respective coordinates for each facial element
			right_eye_curr = face_shape_curr[36:42]
			left_eye_curr = face_shape_curr[42:48]
			mouth_curr = face_shape_curr[60:68]	
			face_curr = face_shape_curr[0:27]
		
		# Open the correct annotation file based on the current frame count
		if (frameCount < 10):
			fileName = (annotPath + "/Frame - 000" + str(frameCount) + ".txt")
		elif (frameCount > 9 and frameCount < 100):
			fileName = (annotPath + "/Frame - 00" + str(frameCount) + ".txt")
		elif (frameCount > 99 and frameCount < 1000):
			fileName = (annotPath + "/Frame - 0" + str(frameCount) + ".txt")
		elif (frameCount > 999 and frameCount < 10000):
			fileName = (annotPath + "/Frame - " + str(frameCount) + ".txt")
	
		file = open(fileName, "r")
	
		# Set the respective annotated coordinates for each facial element
		re_annotations = []
		le_annotations = []
		m_annotations = []
		f_annotations = []
		for i, line in enumerate(file):
			if i in range(0,6):
				re_annotations.append(line)
			elif i in range(7,13):
				le_annotations.append(line)
			elif i in range(14,22):
				m_annotations.append(line)
			elif i in range(23,51):
				f_annotations.append(line)
	
		right_eye_annotations = []
		left_eye_annotations = []
		mouth_annotations = []
		face_annotations = []
	
		testResults.write("Feature Detection Accuracy For Frame " + str(frameCount) + "\n")

		# Divide by scaling ratio to transform frames co-ordinates to match input frames
		for aEntry in re_annotations:
			xCoOr,yCoOr = aEntry.split(",")
			xCoOr = float(xCoOr)/scalingRatio
			yCoOr = float(yCoOr)/scalingRatio
			aLine = [xCoOr, yCoOr]
			right_eye_annotations.append(aLine)
		for aEntry in le_annotations:
			xCoOr,yCoOr = aEntry.split(",")
			xCoOr = float(xCoOr)/scalingRatio
			yCoOr = float(yCoOr)/scalingRatio
			aLine = [xCoOr, yCoOr]
			left_eye_annotations.append(aLine)
		for aEntry in  m_annotations:
			xCoOr,yCoOr = aEntry.split(",")
			xCoOr = float(xCoOr)/scalingRatio
			yCoOr = float(yCoOr)/scalingRatio
			aLine = [xCoOr, yCoOr]
			mouth_annotations.append(aLine)
		for aEntry in  f_annotations:
			xCoOr,yCoOr = aEntry.split(" ")
			xCoOr = float(xCoOr)/scalingRatio
			yCoOr = float(yCoOr)/scalingRatio
			aLine = [xCoOr, yCoOr]
			face_annotations.append(aLine)
		
		# Circle each annotated point 
		for (x, y) in right_eye_annotations:
			cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

		for (x, y) in left_eye_annotations:
			cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

		for (x, y) in mouth_annotations:
			cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

		for (x, y) in face_annotations:
			cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

		totalRSimil = 0
		totalLSimil = 0
		totalMSimil = 0
		totalFSimil = 0
		
		# Compare detected vs annotated coordinates for each point of the right eye element 
		for i in range(len(right_eye_curr)):
			curr_x = right_eye_curr[i][0]
			curr_y = right_eye_curr[i][1]
			annot_x = right_eye_annotations[i][0]
			annot_y = right_eye_annotations[i][1]
			
			if (curr_x == 0 and curr_y == 0):
				xRSimil = 0
				yRSimil = 0
			else:
				xRSimil = (scaledWidth-abs(curr_x-annot_x))/scaledWidth
				yRSimil = (scaledHeight-abs(curr_y-annot_y))/scaledHeight
			totalRSimil += xRSimil + yRSimil
		rSimil = (totalRSimil/12)*100
		finalREye += rSimil
		testResults.write("Right Eye: " + str(rSimil) + "%\n")

		# Compare detected vs annotated coordinates for each point of the left eye element
		for i in range(0, len(left_eye_curr)):
			curr_x = left_eye_curr[i][0]
			curr_y = left_eye_curr[i][1]
			annot_x = left_eye_annotations[i][0]
			annot_y = left_eye_annotations[i][1]

			if (curr_x == 0 and curr_y == 0):
				xLSimil = 0
				yLSimil = 0
			else:
				xLSimil = (scaledWidth-abs(curr_x-annot_x))/scaledWidth
				yLSimil = (scaledHeight-abs(curr_y-annot_y))/scaledHeight
			totalLSimil += xLSimil + yLSimil 
		lSimil = (totalLSimil/12)*100
		finalLEye += lSimil
		testResults.write("Left Eye: " + str(lSimil) + "%\n")
		
		# Compare detected vs annotated coordinates for each point of the mouth element
		for i in range(0, len(mouth_curr)):
			curr_x = mouth_curr[i][0]
			curr_y = mouth_curr[i][1]
			annot_x = mouth_annotations[i][0]
			annot_y = mouth_annotations[i][1]

			if (curr_x == 0 and curr_y == 0):
				xMSimil = 0
				yMSimil = 0
			else:
				xMSimil = (scaledWidth-abs(curr_x-annot_x))/scaledWidth
				yMSimil = (scaledHeight-abs(curr_y-annot_y))/scaledHeight
			totalMSimil += xMSimil + yMSimil
		mSimil = (totalMSimil/16)*100
		finalMouth += mSimil
		testResults.write("Mouth: " + str(rSimil) + "%\n")
		
		# Compare detected vs annotated coordinates for each point of the facial outline element
		for i in range(len(face_curr)):
			curr_x = face_curr[i][0]
			curr_y = face_curr[i][1]
			annot_x = face_annotations[i][0]
			annot_y = face_annotations[i][1]

			if (curr_x == 0 and curr_y == 0):
				xFSimil = 0
				yFSimil = 0
			else:
				xFSimil = (scaledWidth-abs(curr_x-annot_x))/scaledWidth
				yFSimil = (scaledHeight-abs(curr_y-annot_y))/scaledHeight
			totalFSimil += xFSimil + yFSimil
		FSimil = (totalFSimil/56)*100
		finalFace += FSimil
		
		print(frameCount)
		imagename = "Test_Image_" + str(frameCount) +".jpeg"
		cv2.imwrite(imagename, frame)

		frameCount += 1
	
		key = cv2.waitKey(1) & 0xFF
 
		if key == ord("q"):
			break
	
	# Calculate similarities percentages
	finalREye = finalREye/frameCount
	finalLEye = finalLEye/frameCount
	finalMouth = finalMouth/frameCount
	finalFace = finalFace/frameCount

	fullREyeSimil += finalREye
	fullLEyeSimil += finalLEye
	fullMSimil += finalMouth
	fullFSimil += finalFace
	
	# Write results to individual video sample test results and to full test run results files
	testResults.write("Final Test Accuracy: \n" + "Right Eye: " + str(finalREye) + "\nLeft Eye: " + str(finalLEye) + "\nMouth: " + str(finalMouth) + "\nFace Outline: " + str(finalFace))
	print("Final Test Accuracy: \n" + "Right Eye: " + str(finalREye) + "\nLeft Eye: " + str(finalLEye) + "\nMouth: " + str(finalMouth) + "\nFace Outline: " + str(finalFace))
	fullTestResults.write("\nFile " + str(vidNum) + " Test Accuracy: \n" + "Right Eye: " + str(finalREye) + "\nLeft Eye: " + str(finalLEye) + "\nMouth: " + str(finalMouth) + "\nFace Outline: " + str(finalFace))
	vidNum += 1

fullREyeSimil = fullREyeSimil/(vidNum-1)
fullLEyeSimil = fullLEyeSimil/(vidNum-1)
fullMSimil = fullMSimil/(vidNum-1)

fullTestResults.write("\nFull Test Accuracy: \n" + "Right Eye: " + str(fullREyeSimil) + "\nLeft Eye: " + str(fullLEyeSimil) + "\nMouth: " + str(fullMSimil) + "\nFace Outline: " + str(fullFSimil))
print("Full Test Accuracy: \n" + "Right Eye: " + str(fullREyeSimil) + "\nLeft Eye: " + str(fullLEyeSimil) + "\nMouth:" +str(fullMSimil) + "\nFace Outline: " + str(fullFSimil))

# Release the video sample and close all windows
vc.release()
cv2.destroyAllWindows()

