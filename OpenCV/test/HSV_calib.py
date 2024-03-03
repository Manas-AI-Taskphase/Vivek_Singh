import cv2 as cv2
import numpy as np

#cap = cv2.VideoCapture("volleyball_match.mp4")


def nothing():	
	pass

cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

detector = cv2.createBackgroundSubtractorKNN()

while True:
	#ret, frame = cap.read()
	#if not ret:
		#break	
	frame = cv2.imread("test/train2.png")
 
	#blur = cv2.GaussianBlur(frame, (5, 5), sigmaX = 13)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
	lower_h = cv2.getTrackbarPos("LH", "Tracking")
	lower_s = cv2.getTrackbarPos("LS", "Tracking")
	lower_v = cv2.getTrackbarPos("LV", "Tracking")
  
	upper_h = cv2.getTrackbarPos("UH", "Tracking")
	upper_s = cv2.getTrackbarPos("US", "Tracking")
	upper_v = cv2.getTrackbarPos("UV", "Tracking")

	lower_ball = np.array([lower_h, lower_s, lower_v])
	upper_ball = np.array([upper_h, upper_s, upper_v])

	mask = cv2.inRange(hsv, lower_ball, upper_ball)
	res = cv2.bitwise_and(frame, frame, mask=mask)
	
	cv2.imshow("frame", frame)	
	cv2.imshow("mask", mask)
	cv2.imshow("res", res)
 
	key = cv2.waitKey(5) & 0xFF
	if key == ord('q'):
		break
	if key == ord('p'):
		cv2.waitKey(-1)

cv2.destroyAllWindows()
