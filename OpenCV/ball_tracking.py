import cv2
import numpy as np

# Define the lower and upper boundaries by calibrating in ./test
hsv_lower = np.array([9, 121, 126])
hsv_upper = np.array([44, 255, 210])

yuv_lower = np.array([51, 65, 139])
yuv_upper = np.array([227, 102, 182])

# Load the video file
capture = cv2.VideoCapture('volleyball_match.mp4')

detector = cv2.createBackgroundSubtractorKNN()

while True:
    # Capture frame-by-frame
    ret, frame = capture.read()

    if not ret: break
    
    # Convert BGR to HSV
    blur = cv2.GaussianBlur(frame, (5, 5), sigmaX = 13)
    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    yuv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2YUV)

    # Threshold the HSV image to get only yellow colors
    mask1 = cv2.inRange(hsv_frame, hsv_lower, hsv_upper)
    mask2 = cv2.inRange(yuv_frame, yuv_lower, yuv_upper)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Get the radius and center of the contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)

        # Draw the circle around the contour if its radius is greater than a certain value
        if radius < 6 and radius > 0.1:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, 'ball', (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
            
        if radius < 100 and radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, 'player', (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
