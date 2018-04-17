
from scipy.spatial import distance as dist
from scipy import ndimage
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import smbus
import time

# Initialising the face and facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Number of frames processed as assessment frames for assessming normal EAR, MAR, FAR & FC values
assessment_frames = 1800

# Acceptable levels of PERCLOS, Yawn Rate and Drop Rate
acceptable_PERLCOS = ((17/60)*0.4)
acceptable_Yawn = 0.1 
acceptable_Drop = 0.2

# Begin capturing from Pi Camera
print("Camera Warming Up...")
vs = VideoStream(usePiCamera = 1 > 0 ).start()
time.sleep(1.0)

# Function for calculating the FAR and FC given the coordinates for the facial outline
def FAR_FC(facial_landmarks):
	
	global FAR
	resultsArray = []
	
	# Calculate the distance A, B and C for FAR calculation 
	distA = dist.euclidean(facial_landmarks[19], facial_landmarks[6])
	distB = dist.euclidean(facial_landmarks[24], facial_landmarks[10])
 
	distC = dist.euclidean(facial_landmarks[0], facial_landmarks[17])
 
	resultFAR = (distA + distB) / (2.0 * distC)
	
	# Calculate facial centroid using center of mass function 
	resultFC = resultndimage.measurements.center_of_mass(facial_points)
	
	# Append each result to the result array and return when function is called
	resultsArray.append(resultFAR)
	resultsArray.append(resultFC)
	
	return resultsArray

# Function for calculating the EAR given the coordinates for the eye
def EAR(eye_landmarks):
	
	# Calculate the distance A, B and C for EAR calculation
	global EAR
	distA = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
	distB = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
 
	distC = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
 
	resultEAR = (distA + distB) / (2.0 * distC)
	
	# Return resulting EAR value
	return resultEAR

# Function for calculating the MAR given the coordinates for the mouth
def MAR(mouth_landmarks):
		
		# Calculate the distance A, B and C for MAR calculation
        global MAR
        distA = dist.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        distB = dist.euclidean(mouth_landmarks[3], mouth_landmarks[5])
 
        distC = dist.euclidean(mouth_landmarks[0], mouth_landmarks[4])
 
        resultMAR = (distA + distB) / (2.0 * distC)
 
        return resultMAR

# Function for calcualting the normal EAR, MAR, FAR and FC values during the assessment period
def normalEAR_MAR_FAR_FC():
	
	global EAR, totalEAR, MAR, totalMAR, FAR, totalFAR
	frameCount = 0
	totalEAR = 0
	totalMAR = 0
	totalFAR = 0
	totalFC_X = 0
	totalFC_Y = 0
	resultArray = []
	while frameCount <= assessment_frames:
		
		# Query the lux sensor for the current lux level
		bus.write_byte_data(0x39, 0x00 | 0x80, 0x03)
		bus.write_byte_data(0x39, 0x01 | 0x80, 0x02)
		data = bus.read_i2c_block_data(0x39, 0x0C | 0x80, 2)
		data1 = bus.read_i2c_block_data(0x39, 0x0E | 0x80, 2)
		luxLevel = data[1] * 256 + data[0]
		
		# Read in the current frame and resize for processing
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# Convert to grayscale for processing
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# If lux level is below 10 lux perform image processing on frame
		if (luxLevel < 10 ):
			clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(8,8))
			gray = clahe.apply(gray)
			gray = cv2.medianBlur(gray,13)

			imgframe = Image.fromarray(gray)
			enhancer_object = ImageEnhance.Contrast(imgframe)
			out = enhancer_object.enhance(500)
			gray = np.array(out)
		
		# Run face detector on frame
		faces = detector(gray, 0)
		
		# Run the facial landmark detector on the face region returned by previous face detector
		for face in faces:
			face_shape = predictor(gray, face)
			face_shape = face_utils.shape_to_np(face_shape)
			
			# Set the respective coordinates for each facial element
			right_eye = face_shape[36:42]
			left_eye = face_shape[42:48]
			mouth = face_shape[60:68]
			face_outline = face_shape[0:27]
			
			# Calculate the EAR for each eye and the MAR, FAR and FC
			lEAR = EAR(left_eye)
			rEAR = EAR(right_eye)
			rMAR = MAR(mouth)
			rFAR_FC = FAR_FC(face_outline)
			rFAR = rFAR_FC[0]
			rFC = rFAR_FC[1]
						
			avrgEAR = (lEAR + rEAR) / 2.0
			FCxCoOr,FCyCoOr = rFC.split(",")
			
			# Add current values to total for calculating averages
			totalEAR += avrgEAR 
			totalMAR += rMAR
			totalFAR += rFAR
			totalFC_Y += FCyCoOr
			frameCount += 1
		
		# Display the current frame
		cv2.imshow("Normal EAR, MAR & FC Calculation", frame)
		key = cv2.waitKey(1) & 0xFF
	
	# Calculate the respective average values and append to the results array which will be returned when function is called
	finalEAR = totalEAR/assessment_frames
	finalMAR = totalMAR/assessment_frames
	finalFAR = totalFAR/assessment_frames
	finalFC_Y = totalFC_Y/assessment_frames
		
	resultArray.append(finalEAR)
	resultArray.append(finalMAR)
	resultArray.append(finalFAR)
	resultArray.append(finalFC_Y)
	
	# Close current window
	cv2.destroyAllWindows()
	
	return resultArray

print("Calculating normal EAR, MAR, FAR & FC values...")

# Call the function to calculate the respective average values and assign each one to their respective variable
averageEAR_MAR_FAR_FC = normalEAR_MAR_FAR_FC()
averageEAR = averageEAR_MAR_FAR_FC[0]
averageMAR = averageEAR_MAR_FAR_FC[1]
averageFAR = averageEAR_MAR_FAR_FC[2]
averageFC_Y = averageEAR_MAR_FAR_FC[3]

# Print the average values to console
print("Normal EAR is: ", averageEAR)
print("Normal MAR is: ", averageMAR)
print("Normal FAR is: ", averageFAR)
print("Normal FC Y is: ", averageFC_Y)

# Calculate EAR threshold under which eye closure is deemed to have occurred
closureEAR = (averageEAR * 0.5)

# MAR threshold over which yawn is deemed to have occurred
yawnMAR = (averageMAR * 7)

# FAR & FC values to detect head drop
dropFAR = (averageFAR * 0.8)
dropFC = (averageFC_Y + 0.05)

assessClosureArray = []
assessYawnArray = []
assessDropArray = []

for x in range(0, assessment_frames):
	assessClosureArray.append(0)
	assessYawnArray.append(0)
	assessDropArray.append(0)

print("Running fatigue detection algorithm...")

# Repeat while frames available for processing
assessCount = 0
while True:
	
	# Query the lux sensor for the current lux level
	bus.write_byte_data(0x39, 0x00 | 0x80, 0x03)
	bus.write_byte_data(0x39, 0x01 | 0x80, 0x02)
	data = bus.read_i2c_block_data(0x39, 0x0C | 0x80, 2)
	data1 = bus.read_i2c_block_data(0x39, 0x0E | 0x80, 2)
	luxLevel = data[1] * 256 + data[0]
	
	# Read in current frame and resize
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	
	# Convert to grayscale for processing
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# If lux level is below 10 lux perform image processing on frame
	if (luxLevel < 10 ):
		clahe = cv2.createCLAHE(clipLimit=32.0, tileGridSize=(8,8))
		gray = clahe.apply(gray)
		gray = cv2.medianBlur(gray,13)
		imgframe = Image.fromarray(gray)
		enhancer_object = ImageEnhance.Contrast(imgframe)
		out = enhancer_object.enhance(500)
		gray = np.array(out)
	
	# Run face detector on frame
	faces = detector(gray, 0)
	
	# Run the facial landmark detector on the face region returned by previous face detector
	for face in faces:
		face_shape_curr = predictor(gray, face)
		face_shape_curr = face_utils.shape_to_np(face_shape_curr)
		
		# Set the respective coordinates for each facial element
		right_eye_curr = face_shape_curr[36:42]
		left_eye_curr = face_shape_curr[42:48]
		mouth_curr = face_shape_curr[60:68]
		face_outline_curr = face_shape_curr[0:27]
		
		# Calculate the resptive current values 
		lEARCurr = EAR(left_eye_curr)
		rEARCurr = EAR(right_eye_curr)
		rMARCurr = MAR(mouth_curr)
		rFAR_FCCurr = FAR_FC(face_outline)
		rFARCurr = rFAR_FCCurr[0]
		rFCCurr = rFAR_FCCurr[1]
						
		avrgEAR = (lEAR + rEAR) / 2.0
		FCxCoOr,FCyCoOr = rFCCurr.split(",")
		
		avrgEARCurr = (lEARCurr + rEARCurr) / 2.0
		
		# If current EAR is less than threshold set current entry in the eye closure assessment array to 1
		if (avrgEARCurr <= closureEAR):
			assessClosureArray[assessCount] = 1
			print("Eye closure detected")
		
		# If current MAR is greater than threshold set current entry in the yawn assessment array to 1
		if (rMARCurr >= yawnMAR):
			assessYawnArray[assessCount] = 1
			print("Yawn detected")
		
		# If current FAR is less than head drop FAR threshold and then current FC is greater than head drop FC
		if (rFARCurr <= dropFAR):
			if (rFCCurr >= dropFC): 
				assessDropArray[assessCount] = 1
				print("Head drop detected")
	
	# Sum all eye closures and calculate PERLCOS
	eye_closures = sum(int(i) for i in assessClosureArray)
	PERLCOS = eye_closures/assessment_frames
	
	# Sum all yawns and calculate yawn rate
	totalYawn = sum(int(i) for i in assessYawnArray)
	yawnRate = totalYawn/assessment_frames
	
	# Sum all head drops and calculate head drop rate
	totalDrop = sum(int(i) for i in assessDropArray)
	dropRate = totalDrop/assessment_frames
	
	# If current PERCLOS is greater than acceptable PERLCOS, print warning to console
	if (PERLCOS > acceptable_PERLCOS):
#		cv2.putText(frame,'Warning Hazardous Fatigue Levels',(45,235), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
		print("Fatigue Warning Triggered")
	
	# If current yawn rate is greater than acceptable yawn rate, print warning to console
	if (yawnRate > acceptable_Yawn):
#		cv2.putText(frame,'Warning Hazardous Fatigue Levels',(45,235), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
		print("Fatigue Warning Triggered")
	
	# If current drop rate is greater than acceptable drop rate, print warning to console
	if (dropRate > acceptable_Drop):
#		cv2.putText(frame,'Warning Hazardous Fatigue Levels',(45,235), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
		print("Fatigue Warning Triggered")
	
	# Increment frame count if less than the length of assessment period, if equal to length of assessment period then set to 0
	if (assessCount == (int(assessment_frames) - 1)):
		assessCount = 0
	else:
		assessCount += 1
	
	# Display current frame
	cv2.imshow("Fatigue Detection", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Release video capture and destroy all windows
cv2.destroyAllWindows()
vs.stop()
