import torch
import cv2
import numpy as np
from PIL import Image
from model import CRNN
import torchvision.transforms as transforms

# Define the path for the test image and the saved model
test_image_path = "C:/Users/DELL/PycharmProjects/SE173082_ChauMinhNhat/OCR/dataset/img/000061.jpg"
model_path = "crnn_vietnamese_ocr.pth"

class OCRTester:
    def __init__(self, model_path, input_channels=1, output_size=36, hidden_size=256, num_layers=2, image_width=128):
        self.input_channels = input_channels
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.image_width = image_width

        # Initialize the CRNN model
        self.model = CRNN(self.input_channels, self.output_size, self.hidden_size, self.num_layers, self.image_width)

        # Load the trained model weights (set weights_only=True)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()  # Set the model to evaluation mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = Image.fromarray(image)
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        return image

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(image_tensor)

        print(f"Logits shape: {logits.shape}")  # Debug: Check the shape of logits
        logits = logits.permute(1, 0, 2)
        predicted_indices = torch.argmax(logits, dim=2)

        print(f"Predicted indices: {predicted_indices}")  # Debug: Check the predicted indices
        return self.indices_to_text(predicted_indices)

    def indices_to_text(self, indices):
        char_map =  "abcdefghijklmnopqrstuvwxyz0123456789àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ.,? "  # Adjust this based on your dataset
        text = ""
        for idx in indices[0]:  # Assuming batch_size is 1
            if idx > 0 and idx <= len(char_map):
                text += char_map[idx - 1]  # Convert index to character
        return text

    def test_image(self, image_path):
        image_tensor = self.preprocess_image(image_path)
        predicted_text = self.predict(image_tensor)
        return predicted_text

if __name__ == "__main__":
    tester = OCRTester(model_path=model_path)
    predicted_text = tester.test_image(test_image_path)
    print(f"Predicted text: {predicted_text}")