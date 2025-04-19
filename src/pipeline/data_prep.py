import os
import tarfile
import random
import h5py
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader





def get_string(mat, ref):
    """
    Extracts a string from an HDF5 matrix reference.
    Used to get image filenames from the SVHN dataset.
    
    Args:
        mat: HDF5 file object
        ref: Reference to the string data in the HDF5 file
        
    Returns:
        str: The extracted string
    """
    return ''.join([chr(c[0]) for c in mat[ref[0]][()]])


def get_img_boxes(mat, bbox_ref):
    """
    Extracts bounding box information from the SVHN dataset's HDF5 file.
    Each image in SVHN can have multiple digits, each with its own bounding box.
    
    Args:
        mat: HDF5 file object
        bbox_ref: Reference to the bounding box data in the HDF5 file
        
    Returns:
        dict: Dictionary containing lists of height, left, top, width, and label values
              for each digit in the image
    """
    props = ['height', 'left', 'top', 'width', 'label']
    out = {p: [] for p in props}
    box = mat[bbox_ref]
    for p in props:
        arr = box[p]
        if arr.shape[0] == 1:
            out[p].append(int(arr[0][0]))
        else:
            for i in range(arr.shape[0]):
                val_ref = arr[i][0]
                out[p].append(int(mat[val_ref][()].item()))
    return out


def organize_svhn_dataset(root_dir, digit_struct_mat, output_dir, num_bg_per_image=2, bg_iou_thresh=0.1):
    """
    Organizes the SVHN (Street View House Numbers) dataset into class-specific folders.
    SVHN is a real-world image dataset for developing machine learning and object recognition
    algorithms with minimal requirement on data preprocessing and formatting. It contains
    cropped digits from Google Street View images.
    
    This function:
    1. Creates separate directories for each digit (0-9) and background (10)
    2. Processes each image in the dataset
    3. Extracts individual digits using bounding box information
    4. Generates background samples that don't overlap with digits
    5. Saves all cropped images to their respective class directories
    
    Args:
        root_dir (str): Directory containing original SVHN images
        digit_struct_mat (str): Path to digitStruct.mat file containing bounding box info
        output_dir (str): Directory where organized dataset will be saved
        num_bg_per_image (int): Number of background samples to generate per image
        bg_iou_thresh (float): IoU threshold for determining valid background regions
    """
    # Create output directory structure
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class directories (0-9 for digits, 10 for background)
    for i in range(11):
        (output_dir / str(i)).mkdir(exist_ok=True)
    
    # Open HDF5 .mat file
    mat = h5py.File(digit_struct_mat, 'r')
    ds = mat['digitStruct']
    
    # Process digit entries
    n = len(ds['name'])
    for i in tqdm(range(n), desc="Processing images"):
        name = get_string(mat, ds['name'][i])
        bbox_ref = ds['bbox'][i][0]
        bbox = get_img_boxes(mat, bbox_ref)
        img_path = os.path.join(root_dir, name)
        
        # Open the original image
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        
        # Process each digit in the image
        for x, y, w, h, l in zip(bbox['left'], bbox['top'], bbox['width'], bbox['height'], bbox['label']):
            lbl = int(l) % 10
            # Create cropped image
            crop = img.crop((x, y, x+w, y+h)).resize((32, 32), Image.BILINEAR)
            
            # Generate unique filename using image name and coordinates
            base_name = Path(name).stem
            crop_name = f"{base_name}_{x}_{y}_{w}_{h}.png"
            
            # Save to appropriate class directory
            crop.save(output_dir / str(lbl) / crop_name)
        
        # Generate background samples
        boxes = [(x,y,w,h) for x,y,w,h in zip(bbox['left'], bbox['top'], bbox['width'], bbox['height'])]
        for _ in range(num_bg_per_image):
            for _ in range(50):
                cw, ch = 32, 32
                x0 = random.uniform(0, W-cw)
                y0 = random.uniform(0, H-ch)
                
                def iou(b1, b2):
                    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
                    xb = min(b1[0]+b1[2], b2[0]+b2[2]); yb = min(b1[1]+b1[3], b2[1]+b2[3])
                    inter = max(0, xb-xa)*max(0, yb-ya)
                    return inter / (b1[2]*b1[3] + b2[2]*b2[3] - inter + 1e-6)
                
                if all(iou((x0,y0,32,32), bb) < bg_iou_thresh for bb in boxes):
                    # Create background crop
                    bg_crop = img.crop((x0, y0, x0+32, y0+32))
                    bg_name = f"{base_name}_bg_{x0}_{y0}.png"
                    bg_crop.save(output_dir / "10" / bg_name)
                    break
    
    print(f"Dataset organized successfully in {output_dir}")
    print("Class distribution:")
    for i in range(11):
        num_samples = len(list((output_dir / str(i)).glob("*.png")))
        print(f"Class {i}: {num_samples} samples")


class SVHNFull(Dataset):
    """
    PyTorch Dataset class for the organized SVHN dataset.
    This class reads from the organized dataset structure where images are stored in
    class-specific folders (0-9 for digits, 10 for background).
    
    Args:
        root_dir (str): Path to the organized dataset directory
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Collect all samples
        self.samples = []
        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_label = int(class_dir.name)
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, class_label))
        
        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")
        print("Class distribution:")
        for i in range(11):
            num_samples = sum(1 for _, lbl in self.samples if lbl == i)
            print(f"Class {i}: {num_samples} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
if __name__ == "__main__":
    organize_svhn_dataset("data/train", "data/train/digitStruct.mat", "artifacts/data/train")
    organize_svhn_dataset("data/test", "data/test/digitStruct.mat", "artifacts/data/test")