import os
import cv2
import numpy as np
from src.pipeline.inference import Inference

def create_graded_images_folder():
    """Create the graded_images folder if it doesn't exist"""
    if not os.path.exists("graded_images"):
        os.makedirs("graded_images")

def process_image(inference, image_path, output_path):
    """Process a single image and save the result with visualization"""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Get the original image for visualization
    vis_image = image.copy()
    
    # Run inference and get all necessary information
    sequence, predictions, confidences, boxes = inference.inference(image)
    
    # Draw bounding boxes and predictions
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add confidence score for each detection
        idx = np.where(boxes == box)[0][0]
        conf = confidences[idx]
        pred = predictions[idx]
        cv2.putText(vis_image, f"{pred}({conf:.2f})", (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)
    
    # Add the detected sequence as text
    cv2.putText(vis_image, f"Detected: {sequence}", (10, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the result
    cv2.imwrite(output_path, vis_image)
    print(f"Processed {image_path} -> {output_path}")
    print(f"Detected sequence: {sequence}")
    # print(f"Individual predictions: {list(zip(predictions, confidences))}")

def main():
    # Create output directory
    create_graded_images_folder()
    
    # Initialize the inference pipeline
    inference = Inference()
    
    # Process each test image
    for i in range(1, 6):
        input_path = f"images/{i}.png"
        output_path = f"graded_images/{i}.png"
        
        if os.path.exists(input_path):
            process_image(inference, input_path, output_path)
        else:
            print(f"Warning: Image {input_path} not found")

if __name__ == "__main__":
    main()
