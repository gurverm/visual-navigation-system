import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load models
yolo_model = YOLO('yolov8n.pt')

model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Move model to GPU (CPU if GPU is not available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform 

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = yolo_model(frame, verbose=False)
    
    # Preprocess for depth: convert BGR to RGB, apply MiDaS transform
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    # Run depth inference
    with torch.no_grad():
        prediction = midas(input_batch)
        
        # Resize prediction to match original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy
    depth_map = prediction.cpu().numpy()
    
    # Get annotated frame from YOLO
    annotated_frame = results[0].plot()
    
    # Loop through each detected object
    for box in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # TODO 1: Extract depth values
        depth_box = depth_map[y1:y2, x1:x2]

        # TODO 2: Calculate median
        median_depth = np.median(depth_box)

        # TODO 3: Draw text
        cv2.putText(annotated_frame, f"{median_depth:.2f}m", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
    
    # Display
    cv2.imshow('Detection + Depth', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()