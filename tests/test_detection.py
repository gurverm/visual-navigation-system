import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (smallest/fastest version)
# Other options: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
model = YOLO('yolov8n.pt')

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    # verbose=False stops it from printing to console every frame
    results = model(frame, verbose=False)
    
    # results[0] contains all detections for this frame
    # .plot() draws boxes and labels on the frame
    annotated_frame = results[0].plot()
    
    # Display
    cv2.imshow('Object Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()