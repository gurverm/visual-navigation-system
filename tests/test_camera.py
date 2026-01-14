import cv2

# Open the webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press 'q' to quit")

while True:
    # Read frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Display the frame
    cv2.imshow('Camera Test', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()