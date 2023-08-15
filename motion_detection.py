
import cv2
import numpy as np

# Initialize video capture from default camera
cp=cv2.VideoCapture(0)


first_frame=None  

while True:
    _,frame=cp.read(0)
    
    # Convert the frame to grayscale and apply Gaussian blur
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    
    if first_frame is None:
        first_frame=gray
        continue
        
    # Checking if there is any change in pixel values of frame and setting a threshold for the change    
    delta_frame=cv2.absdiff(first_frame,gray)
    threshold=cv2.threshold(delta_frame,50,255,cv2.THRESH_BINARY)[1]
    threshold_frame=cv2.dilate(threshold,None,iterations=2)
    
     # Find contours in the thresholded frame
    (cntr,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in cntr:
        if cv2.contourArea(contour)<1000: # If the contour area is smaller than a threshold, skip it
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
          
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cp.release
cv2.destroyAllWindows()
