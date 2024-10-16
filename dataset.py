# dataset.py
import os
import json
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Define the character set
CHARACTER_SET = "abcdefghijklmnopqrstuvwxyz0123456789àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ "

# Create a mapping from character to index and reverse mapping
char_to_index = {char: idx for idx, char in enumerate(CHARACTER_SET)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
class VietnameseOCRDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg'))]

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in directory {image_folder}")

        print(f"Number of images found: {len(self.image_files)}")

        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.image_files[idx].replace('.jpg', '.json'))

        # Ensure the label file exists
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file for {self.image_files[idx]} not found.")

        # Load image using OpenCV (this gives a numpy array)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert numpy array (OpenCV image) to PIL Image
        image = Image.fromarray(image)

        # Apply the transformation
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
            text_label = label_data.get('text', '')

        return image, text_label

    def __len__(self):
        return len(self.image_files)