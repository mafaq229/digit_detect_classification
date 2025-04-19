import cv2


def get_roi_images(image, boxes, target_size=(32, 32)):
    """
    Extracts and optionally resizes regions of interest (digit ROIs) from the image.
    
    Args:
        image (np.array): The original image.
        boxes (np.array): Array of bounding boxes [x1, y1, x2, y2].
        target_size (tuple): The size to which each ROI is resized.
    
    Returns:
        List of ROI images.
    """
    rois = []
    for (x1, y1, x2, y2) in boxes:
        roi = image[y1:y2, x1:x2]
        # Optionally, convert to grayscale
        # if len(roi.shape) == 3:
        #     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Resize ROI to the target size (e.g., what the classifier expects)
        roi = cv2.resize(roi, target_size)
        rois.append(roi)
    return rois

