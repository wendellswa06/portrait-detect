import cv2
import os
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

def detect_and_save_persons(image_path, model_path="yolov8l.pt", conf_threshold=0.5):
    
    # Create output directory
    output_dir = "detected_faces"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="./model.pt")

    # Load the YOLO model
    model = YOLO(model_path)
    model.conf = conf_threshold
    
    # Read the image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    print(height, width)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return 0
    
    # Convert BGR to RGB (YOLO expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(image_rgb)
    
    # Create a copy for drawing all detections
    annotated_image = image_rgb.copy()
    
    # Initialize person counter
    count = 0
    
    # Extract base filename without extension for naming the output files
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            count += 1
                          
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
            # Extract the person from the original image
            y1 = 0 if y1 -50 < 0 else y1 - 50
            x1 = 0 if x1 -50 < 0 else x1 - 50
            x2 = width if x2 + 50 > width else x2 + 50
            y2 = height if y2 + 50 > height else y2 + 50
            
            face_image = image[y1:y2, x1:x2]
            
            # Save the extracted person
            output_path = os.path.join(output_dir, f"face.jpg")
            if count == 1:
                cv2.imwrite(output_path, face_image)
            print(f"Saved face {count} to {output_path}")
    
    # Print results
    
    return count

if __name__ == "__main__":
    # Example usage
    image_path = "5.png"  # Replace with your image path
    count = detect_and_save_persons(image_path)
    print(count)

