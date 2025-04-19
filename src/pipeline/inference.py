"""Inference pipeline for the model"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from src.components.mesr import MESR
from src.components.models import VGG16FineTune, inference_transform
from src.utils import get_roi_images


class Inference:
    def __init__(self):
        self.mesr = MESR()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "artifacts/models/vgg16_svhn_full_11cls_6_epoch.pth"
        self.transform = inference_transform
    
    def get_rois(self, image):
        """Get the ROIs from the image"""
        boxes = self.mesr.apply(image)
        rois = get_roi_images(image, boxes, target_size=(32, 32))
        return rois, boxes
    
    def inference(self, image):
        """Inference the image"""
        rois, boxes = self.get_rois(image)
        if not rois:
            return [], []
        
        # Convert ROIs to PIL Images and then to tensors
        roi_tensors = []
        for roi in rois:
            # Convert BGR to RGB and numpy array to PIL Image
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(roi_rgb)
            # Apply transforms and move to device
            tensor = self.transform(pil_image).to(self.device)
            roi_tensors.append(tensor)
        
        # Stack tensors along the batch dimension
        batch = torch.stack(roi_tensors)
        
        # Load and prepare model
        model = VGG16FineTune(num_classes=11).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        # Perform inference
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = probs.max(dim=1)
            
        # Convert predictions to digit classes (0-9, 10 for background)
        predictions = predictions.cpu().numpy()
        confidences = confidences.cpu().numpy()
        
        output_sequence = self.postprocess(predictions, confidences, boxes)

        return output_sequence
    
    def postprocess(self, predictions, confidences, boxes):
        """Postprocess the predictions to get the final output of digits in a sequence"""

        # Discard the background category (class 10)
        valid_mask = predictions != 10
        predictions = predictions[valid_mask]
        confidences = confidences[valid_mask]
        boxes = boxes[valid_mask]
        
        if len(predictions) == 0:
            return ""
            
        # Discard predictions with low confidence
        confidence_threshold = 0.7
        high_conf_mask = confidences >= confidence_threshold
        predictions = predictions[high_conf_mask]
        confidences = confidences[high_conf_mask]
        boxes = boxes[high_conf_mask]
        
        if len(predictions) == 0:
            return ""
            
        # Calculate box centers for sorting
        centers = np.array([(box[0] + box[2]) / 2 for box in boxes])
        
        # Sort predictions from left to right based on box centers
        sorted_indices = np.argsort(centers)
        predictions = predictions[sorted_indices]
        confidences = confidences[sorted_indices]
        boxes = boxes[sorted_indices]
        
        # Convert predictions to string
        sequence = ''.join([str(pred) for pred in predictions])
        
        return sequence, predictions, confidences, boxes
    
if __name__ == "__main__":
    inference = Inference()
    image = cv2.imread("images/3.png")
    
    output_sequence, _, _, _ = inference.inference(image)
    print(output_sequence)
    
    
    # boxes = inference.inference(image)
    # # Optionally, draw bounding boxes for visualization.
    # vis = image.copy()
    # for (x1, y1, x2, y2) in boxes:
    #     cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow("Detected Digit Regions", vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # # Extract and preprocess ROIs (resized, grayscale) to feed into your PyTorch model.
    # rois = get_roi_images(image, boxes)
    # # For demonstration: show each ROI.
    # for idx, roi in enumerate(rois):
    #     cv2.imshow(f"ROI {idx}", roi)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(boxes)