import cv2
import torch
import numpy as np

# load the model
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

# move model to gpu (cpu if gpu is not an option)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# load the preprocessing that midas expects 
midas_transforms = torch.hub.load("intel-isl/MiDaS","transforms")
transform = midas_transforms.dpt_transform 

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess: convert BGR (OpenCV color format) to RGB, apply MiDaS transform
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    # Run inference
    with torch.no_grad():  # Don't compute gradients (we're not training)
        prediction = midas(input_batch)
        
        # Resize prediction to match original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy for visualization
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth to 0-255 for display
    ## what exactly does this normailze mean? Play around with docs 
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)
    
    # Apply colormap (makes it easier to see - far is blue, close is red)
    ## play around with the different color schemes -- check docs
    depth_colormap = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_MAGMA)
    
    # Display original and depth side by side
    combined = np.hstack((frame, depth_colormap))
    cv2.imshow('Camera | Depth', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()