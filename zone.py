#without supervision

from ultralytics import YOLO
import cv2
from urllib.request import urlopen


url = 'http://192.168.0.100/cam-hi.jpg'

cap=cv2.VideoCapture(url)

model = YOLO('yolov8n.pt')

count=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    frame=cv2.resize(frame,(640,480))

    roi = frame[0:480,0:320]

    results = model(roi, classes=0)

    human_detected = False

    for result in results:  # Iterate through each result
        boxes = result.boxes.xyxy  # Get bounding boxes (x1, y1, x2, y2)
        classes = result.boxes.cls  # Get class indices

        for i in range(len(boxes)):
            # Get the coordinates and class for each detected object
            x1, y1, x2, y2 = boxes[i]
            class_id = int(classes[i])  # Convert to integer for class ID

            # Check if the detected bounding box is within the ROI
            if (class_id == 0):  # Check if the class is 'person' (usually class 0)
                human_detected = True
        
        cv2.rectangle(roi,(0,480),(320,0),(0, 255, 0),2)
        

    if human_detected:
        print("Human detected - Relay ON")
        relay_status = True  # Fake GPIO output for local testing
    else:
        relay_status = False  # Fake GPIO off

    cv2.imshow("ROI",frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
