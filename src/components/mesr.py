"""Code for the MESR component"""

import numpy as np
import cv2


class MESR:
    def __init__(self, min_area=80, max_area=2000):
        """Initialize MESR with parameters
        
        Default parameters to MSER_create are:
            delta (int): Delta parameter. Default 5
            min_area (int): Minimum area threshold. Default 60 
            max_area (int): Maximum area threshold. Default 14400
            max_variation (float): Maximum variation allowed. Default 0.25
            min_diversity (float): Minimum diversity required. Default 0.2
            max_evolution (int): Maximum evolution steps. Default 200
            area_threshold (float): Area threshold. Default 1.01
            min_margin (float): Minimum margin. Default 0.003
            edge_blur_size (int): Edge blur kernel size. Default 5
        """
        self.min_area = min_area
        self.max_area = max_area
        self.mesr = cv2.MSER_create(min_area=min_area, max_area=max_area)
    
    def preprocess(self, image):
        """Preprocess the image for MESR
        
        Args:
            image (numpy.ndarray): Input image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur the image
        # blurred = cv2.GaussianBlur(gray, (3, 3), 1)
        # Apply bilateral filter to reduce noise while preserving edges
        bilateral_gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        return bilateral_gray
    
    def filter_regions(self, boxes, aspect_ratio_thresholds=(0.2, 1.2)):
        """Filter regions based on aspect ratio and apply non-maximum suppression
        
        Args:
            boxes (list): List of bounding boxes in [x, y, w, h] format
            aspect_ratio_thresholds (tuple): Aspect ratio thresholds for filtering
        """
        
        #TODO: Fix the code since mesr returns boxes as well. check if we can use them
        filtered_boxes = []
        for box in boxes:
            x, y, w, h = box
            aspect_ratio = w / h
            area = w * h
            # Check if aspect ratio is within thresholds
            if aspect_ratio < aspect_ratio_thresholds[0] or aspect_ratio > aspect_ratio_thresholds[1]:
                continue
            # Check if area is within thresholds
            if area < self.min_area or area > self.max_area:
                continue
            filtered_boxes.append([x, y, w, h])

        filtered_boxes = np.array(filtered_boxes)
        #TODO: apply nms before returning the boxes
        filtered_boxes = self.non_max_suppression(filtered_boxes)
        return filtered_boxes
    
    def non_max_suppression(self, boxes, overlapThresh=0.3):
        """Applies non-maximum suppression to bounding boxes."""
        # Using area as score i.e. larger boxes get higher priority
        scores = boxes[:, 2] * boxes[:, 3]
        
        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  score_threshold=0.0,
                                  nms_threshold=overlapThresh)
        return boxes[indices]
    
    def convert_to_original_coordinates(self, boxes):
        """Convert boxes to original image coordinates"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x2 = x1 + w
        y2 = y1 + h
        return np.column_stack((x1, y1, x2, y2))
    
    def apply(self, image):
        """Apply MESR to an image
        
        Args:
            image (numpy.ndarray): Input image
        """
        # Preprocess the image
        preprocessed_image = self.preprocess(image)
        # Detect regions
        regions, boxes = self.mesr.detectRegions(preprocessed_image)

        # Filter regions based on aspect ratio and apply non-maximum suppression
        filtered_boxes = self.filter_regions(boxes)

        # Convert boxes to original image coordinates
        filtered_boxes = self.convert_to_original_coordinates(filtered_boxes)
        
        return filtered_boxes
    
    